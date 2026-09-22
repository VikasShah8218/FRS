/*
 * face_engine.cpp
 *
 * Implementation notes / gotchas that are easy to get wrong:
 *
 *  - The two models want DIFFERENT input:
 *      detector (YuNet):   BGR, raw 0..255, no normalisation
 *      recognition:        RGB, (px - 127.5) / 127.5
 *    Mixing them up is a silent accuracy bug. Both are NCHW float32.
 *
 *  - YuNet 2023mar has a FIXED 1x3x640x640 input and 12 outputs,
 *    cls_/obj_/bbox_/kps_ for strides 8, 16 and 32. load() requests them
 *    by name, so their order in the file does not matter.
 *
 *  - One prior per grid cell, row-major: idx = row * cols + col.
 *    cls and obj are already probabilities; score = sqrt(cls * obj).
 *
 *  - bbox = (dx, dy, log w, log h): centre = (col + dx, row + dy) * stride,
 *    size = exp(log w | log h) * stride. It is NOT x1,y1,x2,y2.
 *    kps  = 5 x (dx, dy): point = (col + dx, row + dy) * stride.
 *
 *  - YuNet finds faces of roughly 10-300 px. detect() therefore runs one
 *    pass per entry of detScales and merges the candidates with a single
 *    NMS; a close-up face that is too big at 640 is found at 320 or 160.
 *
 *  - Letterbox: image goes at the TOP-LEFT of the 640x640 zero canvas.
 *    Aspect ratio preserved. Remember the scale to map results back.
 *
 *  - ESSI-FR v1 already outputs L2-normalised embeddings; embed() normalises
 *    again anyway, which is harmless.
 */

#include "face_engine.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace essi {

/* ---- 5-point reference template for a 112x112 crop (the alignment ESSI-FR v1 was trained on) ---- */
static const float kRefPoints[5][2] = {
    {38.2946f, 51.6963f},   /* left eye     */
    {73.5318f, 51.5014f},   /* right eye    */
    {56.0252f, 71.7366f},   /* nose tip     */
    {41.5493f, 92.3655f},   /* left mouth   */
    {70.7299f, 92.2041f}    /* right mouth  */
};

static const int kStrides[3]     = {8, 16, 32};

/* YuNet output tensors, in the order detectAtScale() indexes them. */
static const char *const kDetOutputs[12] = {
    "cls_8",  "cls_16",  "cls_32",
    "obj_8",  "obj_16",  "obj_32",
    "bbox_8", "bbox_16", "bbox_32",
    "kps_8",  "kps_16",  "kps_32"
};

namespace {
/* NIST General spec 7.1: the library must never write to stdout/stderr.
 * ONNX Runtime's default logger prints to stderr, so every ORT message
 * (any severity) is sent here and dropped. */
void ORT_API_CALL silentOrtLogger(void *, OrtLoggingLevel, const char *,
                                 const char *, const char *, const char *)
{
}
}  /* anonymous namespace */

/* ======================================================================= */

FaceEngine::FaceEngine() {}
FaceEngine::~FaceEngine() {}

std::string
FaceEngine::load(const std::string &configDir)
{
    try {
        env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_FATAL, "essi",
                                         silentOrtLogger, nullptr);

        Ort::SessionOptions opts;
        /* NIST times single-threaded. Do not let ORT grab every core. */
        opts.SetIntraOpNumThreads(1);
        opts.SetInterOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        /* Always build the path from configDir. Never hard-code it -
         * the validation script actively checks for hard-coded paths. */
        const std::string detPath = configDir + "/face_detection_yunet_2023mar.onnx";
        const std::string recPath = configDir + "/essi_fr_v1_last.onnx";

        detSession = std::make_unique<Ort::Session>(*env, detPath.c_str(), opts);
        recSession = std::make_unique<Ort::Session>(*env, recPath.c_str(), opts);

        Ort::AllocatorWithDefaultOptions alloc;

        auto grab = [&](Ort::Session &s,
                        std::vector<std::string> &ins,
                        std::vector<std::string> &outs) {
            for (size_t i = 0; i < s.GetInputCount(); i++)
                ins.push_back(s.GetInputNameAllocated(i, alloc).get());
            for (size_t i = 0; i < s.GetOutputCount(); i++)
                outs.push_back(s.GetOutputNameAllocated(i, alloc).get());
        };

        grab(*detSession, detInNames, detOutNames);
        grab(*recSession, recInNames, recOutNames);

        /* Request the detector outputs by name, in kDetOutputs order. */
        for (const char *want : kDetOutputs) {
            if (std::find(detOutNames.begin(), detOutNames.end(), want)
                    == detOutNames.end())
                return std::string("detector has no output named ") + want;
        }
        detOutNames.assign(std::begin(kDetOutputs), std::end(kDetOutputs));

        const std::vector<int64_t> inShape = detSession->GetInputTypeInfo(0)
            .GetTensorTypeAndShapeInfo().GetShape();
        if (inShape.size() != 4 || inShape[2] <= 0 || inShape[2] != inShape[3])
            return "detector input must be 1x3xSxS";
        detInputSize = static_cast<int>(inShape[2]);

    } catch (const Ort::Exception &e) {
        return std::string("onnxruntime: ") + e.what();
    } catch (const std::exception &e) {
        return std::string("load failed: ") + e.what();
    }
    return "";
}

/* ======================================================================= */
/* Detection                                                                */
/* ======================================================================= */

namespace {

float iou(const FaceDet &a, const FaceDet &b)
{
    const float ix1 = std::max(a.x1, b.x1);
    const float iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2);
    const float iy2 = std::min(a.y2, b.y2);
    const float iw  = std::max(0.0f, ix2 - ix1);
    const float ih  = std::max(0.0f, iy2 - iy1);
    const float inter = iw * ih;
    const float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    const float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    const float uni   = areaA + areaB - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

/* Bilinear resize of an interleaved RGB buffer. */
void resizeRGB(const uint8_t *src, int sw, int sh,
               uint8_t *dst, int dw, int dh)
{
    const float rx = static_cast<float>(sw) / dw;
    const float ry = static_cast<float>(sh) / dh;
    for (int y = 0; y < dh; y++) {
        float fy = (y + 0.5f) * ry - 0.5f;
        if (fy < 0) fy = 0;
        int   y0 = static_cast<int>(fy);
        int   y1 = std::min(y0 + 1, sh - 1);
        float wy = fy - y0;
        for (int x = 0; x < dw; x++) {
            float fx = (x + 0.5f) * rx - 0.5f;
            if (fx < 0) fx = 0;
            int   x0 = static_cast<int>(fx);
            int   x1 = std::min(x0 + 1, sw - 1);
            float wx = fx - x0;
            for (int c = 0; c < 3; c++) {
                const float p00 = src[(y0 * sw + x0) * 3 + c];
                const float p01 = src[(y0 * sw + x1) * 3 + c];
                const float p10 = src[(y1 * sw + x0) * 3 + c];
                const float p11 = src[(y1 * sw + x1) * 3 + c];
                const float top = p00 + (p01 - p00) * wx;
                const float bot = p10 + (p11 - p10) * wx;
                float v = top + (bot - top) * wy;
                if (v < 0) v = 0;
                if (v > 255) v = 255;
                dst[(y * dw + x) * 3 + c] = static_cast<uint8_t>(v + 0.5f);
            }
        }
    }
}

}  /* anonymous namespace */

void
FaceEngine::detectAtScale(const uint8_t *rgb, int width, int height,
                          int maxSide, std::vector<FaceDet> &out)
{
    const int S = detInputSize;
    if (maxSide > S) maxSide = S;

    /* --- letterbox: longest side = maxSide, top-left of the SxS canvas --- */
    const float scale = std::min(static_cast<float>(maxSide) / width,
                                 static_cast<float>(maxSide) / height);
    const int nw = std::min(S, std::max(1, static_cast<int>(width  * scale)));
    const int nh = std::min(S, std::max(1, static_cast<int>(height * scale)));

    std::vector<uint8_t> small(static_cast<size_t>(nw) * nh * 3);
    resizeRGB(rgb, width, height, small.data(), nw, nh);

    /* --- to NCHW float, BGR, raw 0..255; the padding stays 0 (black) --- */
    const size_t plane = static_cast<size_t>(S) * S;
    std::vector<float> input(3 * plane, 0.0f);
    for (int y = 0; y < nh; y++) {
        for (int x = 0; x < nw; x++) {
            const uint8_t *p = &small[(static_cast<size_t>(y) * nw + x) * 3];
            const size_t o = static_cast<size_t>(y) * S + x;
            input[o]             = p[2];   /* B */
            input[plane + o]     = p[1];   /* G */
            input[2 * plane + o] = p[0];   /* R */
        }
    }

    try {
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        std::array<int64_t, 4> shape{1, 3, S, S};
        Ort::Value tensor = Ort::Value::CreateTensor<float>(
            mem, input.data(), input.size(), shape.data(), shape.size());

        std::vector<const char *> in{detInNames[0].c_str()};
        std::vector<const char *> names;
        for (auto &n : detOutNames) names.push_back(n.c_str());

        auto outputs = detSession->Run(Ort::RunOptions{nullptr},
                                       in.data(), &tensor, 1,
                                       names.data(), names.size());

        /* outputs follow kDetOutputs: cls[0..2] obj[3..5] bbox[6..8] kps[9..11] */
        for (int lvl = 0; lvl < 3; lvl++) {
            const int stride = kStrides[lvl];
            const int cols   = S / stride;
            const int rows   = S / stride;

            const float *cls = outputs[lvl    ].GetTensorData<float>();
            const float *obj = outputs[lvl + 3].GetTensorData<float>();
            const float *bb  = outputs[lvl + 6].GetTensorData<float>();
            const float *kp  = outputs[lvl + 9].GetTensorData<float>();

            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    const size_t idx = static_cast<size_t>(r) * cols + c;
                    const float cs = std::min(1.0f, std::max(0.0f, cls[idx]));
                    const float os = std::min(1.0f, std::max(0.0f, obj[idx]));
                    const float score = std::sqrt(cs * os);
                    if (score < detThreshold) continue;

                    const float cx = (c + bb[idx * 4 + 0]) * stride;
                    const float cy = (r + bb[idx * 4 + 1]) * stride;
                    const float w  = std::exp(bb[idx * 4 + 2]) * stride;
                    const float h  = std::exp(bb[idx * 4 + 3]) * stride;

                    FaceDet f;
                    f.score = score;
                    f.x1 = (cx - w / 2) / scale;
                    f.y1 = (cy - h / 2) / scale;
                    f.x2 = (cx + w / 2) / scale;
                    f.y2 = (cy + h / 2) / scale;
                    for (int k = 0; k < 5; k++) {
                        f.kps[k][0] = (c + kp[idx * 10 + k * 2 + 0]) * stride / scale;
                        f.kps[k][1] = (r + kp[idx * 10 + k * 2 + 1]) * stride / scale;
                    }
                    out.push_back(f);
                }
            }
        }
    } catch (const Ort::Exception &) {
        /* a failed pass adds nothing; the other scales still count */
    }
}

std::vector<FaceDet>
FaceEngine::detect(const uint8_t *rgb, int width, int height)
{
    std::vector<FaceDet> result;
    if (!detSession || !rgb || width <= 0 || height <= 0) return result;

    /* One pass per scale, all candidates pooled, then one NMS over the pool:
     * a face found at two scales collapses to its best-scoring detection. */
    std::vector<FaceDet> cands;
    for (int maxSide : detScales)
        detectAtScale(rgb, width, height, maxSide, cands);

    /* --- NMS, highest score first --- */
    std::sort(cands.begin(), cands.end(),
              [](const FaceDet &a, const FaceDet &b) { return a.score > b.score; });

    std::vector<bool> dead(cands.size(), false);
    for (size_t i = 0; i < cands.size(); i++) {
        if (dead[i]) continue;
        for (size_t j = i + 1; j < cands.size(); j++) {
            if (!dead[j] && iou(cands[i], cands[j]) > nmsThreshold)
                dead[j] = true;
        }
        result.push_back(cands[i]);
    }
    return result;
}

/* ======================================================================= */
/* Alignment: 2D similarity transform (Umeyama, no reflection)             */
/* ======================================================================= */

void
FaceEngine::warpToTemplate(const uint8_t *rgb, int width, int height,
                           const FaceDet &face, float *outFloat,
                           uint8_t *outBytes)
{
    /* --- least-squares similarity transform: src kps -> reference --- */
    float mpx = 0, mpy = 0, mqx = 0, mqy = 0;
    for (int i = 0; i < 5; i++) {
        mpx += face.kps[i][0];  mpy += face.kps[i][1];
        mqx += kRefPoints[i][0]; mqy += kRefPoints[i][1];
    }
    mpx /= 5; mpy /= 5; mqx /= 5; mqy /= 5;

    float sDot = 0, sCross = 0, varP = 0;
    for (int i = 0; i < 5; i++) {
        const float ax = face.kps[i][0] - mpx, ay = face.kps[i][1] - mpy;
        const float bx = kRefPoints[i][0] - mqx, by = kRefPoints[i][1] - mqy;
        sDot   += ax * bx + ay * by;
        sCross += ax * by - ay * bx;
        varP   += ax * ax + ay * ay;
    }
    if (varP < 1e-9f) varP = 1e-9f;

    const float c = sDot / varP;      /* scale * cos(theta) */
    const float s = sCross / varP;    /* scale * sin(theta) */

    /* forward:  q = [[c,-s],[s,c]] * p + t   */
    const float tx = mqx - (c * mpx - s * mpy);
    const float ty = mqy - (s * mpx + c * mpy);

    /* we need the inverse to sample the source per output pixel */
    const float det = c * c + s * s;
    const float ic =  c / det, is = -s / det;
    /* inverse matrix is [[ic, -is],[is, ic]] with the signs above folded in */
    const float itx = -(ic * tx - is * ty);
    const float ity = -(is * tx + ic * ty);

    for (int v = 0; v < 112; v++) {
        for (int u = 0; u < 112; u++) {
            const float sx = ic * u - is * v + itx;
            const float sy = is * u + ic * v + ity;

            float px[3] = {0, 0, 0};
            if (sx >= 0 && sy >= 0 && sx <= width - 1 && sy <= height - 1) {
                const int   x0 = static_cast<int>(sx);
                const int   y0 = static_cast<int>(sy);
                const int   x1 = std::min(x0 + 1, width  - 1);
                const int   y1 = std::min(y0 + 1, height - 1);
                const float wx = sx - x0, wy = sy - y0;
                for (int ch = 0; ch < 3; ch++) {
                    const float p00 = rgb[(y0 * width + x0) * 3 + ch];
                    const float p01 = rgb[(y0 * width + x1) * 3 + ch];
                    const float p10 = rgb[(y1 * width + x0) * 3 + ch];
                    const float p11 = rgb[(y1 * width + x1) * 3 + ch];
                    const float top = p00 + (p01 - p00) * wx;
                    const float bot = p10 + (p11 - p10) * wx;
                    px[ch] = top + (bot - top) * wy;
                }
            }

            if (outBytes) {
                for (int ch = 0; ch < 3; ch++) {
                    float q = px[ch];
                    if (q < 0) q = 0;
                    if (q > 255) q = 255;
                    outBytes[(v * 112 + u) * 3 + ch] =
                        static_cast<uint8_t>(q + 0.5f);
                }
            }
            if (outFloat) {
                /* NCHW, (px - 127.5) / 127.5  <- note: NOT 128.0 */
                for (int ch = 0; ch < 3; ch++)
                    outFloat[ch * 112 * 112 + v * 112 + u] =
                        (px[ch] - 127.5f) / 127.5f;
            }
        }
    }
}

void
FaceEngine::alignedCrop(const uint8_t *rgb, int width, int height,
                        const FaceDet &face, uint8_t *out112)
{
    warpToTemplate(rgb, width, height, face, nullptr, out112);
}

/* ======================================================================= */
/* Embedding                                                                */
/* ======================================================================= */

std::vector<float>
FaceEngine::embed(const uint8_t *rgb, int width, int height,
                  const FaceDet &face)
{
    std::vector<float> emb;
    if (!recSession || !rgb) return emb;

    std::vector<float> input(static_cast<size_t>(3) * 112 * 112);
    warpToTemplate(rgb, width, height, face, input.data(), nullptr);

    try {
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        std::array<int64_t, 4> shape{1, 3, 112, 112};
        Ort::Value tensor = Ort::Value::CreateTensor<float>(
            mem, input.data(), input.size(), shape.data(), shape.size());

        std::vector<const char *> in{recInNames[0].c_str()};
        std::vector<const char *> out{recOutNames[0].c_str()};

        auto outputs = recSession->Run(Ort::RunOptions{nullptr},
                                       in.data(), &tensor, 1,
                                       out.data(), 1);

        const float *p = outputs[0].GetTensorData<float>();
        const size_t n = outputs[0].GetTensorTypeAndShapeInfo()
                             .GetElementCount();

        emb.assign(p, p + n);

        /* L2 normalize (ESSI-FR v1 already does; repeating it is harmless) */
        double sum = 0.0;
        for (float v : emb) sum += static_cast<double>(v) * v;
        const float norm = static_cast<float>(std::sqrt(sum));
        if (norm > 1e-9f)
            for (float &v : emb) v /= norm;

    } catch (const Ort::Exception &) {
        emb.clear();
    }
    return emb;
}

float
FaceEngine::cosine(const std::vector<float> &a, const std::vector<float> &b)
{
    if (a.size() != b.size() || a.empty()) return 0.0f;
    double d = 0.0;
    for (size_t i = 0; i < a.size(); i++)
        d += static_cast<double>(a[i]) * b[i];
    if (d >  1.0) d =  1.0;
    if (d < -1.0) d = -1.0;
    return static_cast<float>(d);
}

}  /* namespace essi */
