/*
 * face_engine.h
 *
 * Self-contained face detection + alignment + embedding engine.
 * Deliberately has NO NIST/FRVT types in it, so you can unit-test it
 * on its own against your Python oracle before wiring it into the API.
 *
 * Pipeline:  RGB image  ->  YuNet detect (face_detection_yunet_2023mar.onnx,
 *                           OpenCV Zoo, MIT licence) at several scales
 *                       ->  5-point similarity-transform align to 112x112
 *                       ->  ESSI-FR v1 embed (essi_fr_v1_last.onnx)
 *                       ->  512 floats, L2-normalized
 */

#ifndef FACE_ENGINE_H_
#define FACE_ENGINE_H_

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace essi {

/* One detected face. */
struct FaceDet {
    float x1, y1, x2, y2;     /* bounding box, original image coords   */
    float score;              /* detector confidence                   */
    float kps[5][2];          /* 5 landmarks, original image coords    */
                              /* order as seen in the image: left eye,
                               * right eye, nose, left mouth corner,
                               * right mouth corner                    */
};

class FaceEngine {
public:
    FaceEngine();
    ~FaceEngine();

    /* Load both models. configDir is the directory NIST hands you.
     * Returns "" on success, or an error message on failure. */
    std::string load(const std::string &configDir);

    /* Detect every face. Returned sorted by score, highest first.
     * rgb must be width*height*3 bytes, 8-bit, R,G,B interleaved. */
    std::vector<FaceDet> detect(const uint8_t *rgb, int width, int height);

    /* Align one face to 112x112 and run the recognition model.
     * Returns 512 L2-normalized floats. Empty vector on failure. */
    std::vector<float> embed(const uint8_t *rgb, int width, int height,
                             const FaceDet &face);

    /* Debug helper: write the aligned 112x112 crop into out (112*112*3 bytes).
     * Use this to compare against your Python *_aligned.png files. */
    void alignedCrop(const uint8_t *rgb, int width, int height,
                     const FaceDet &face, uint8_t *out112);

    /* Cosine similarity of two L2-normalized 512-vectors. Range -1..1. */
    static float cosine(const std::vector<float> &a, const std::vector<float> &b);

    /* Tunables. */
    float detThreshold = 0.6f;   /* YuNet score = sqrt(cls * obj)          */
    float nmsThreshold = 0.3f;   /* YuNet's reference NMS threshold        */
    /* Longest image side for each detector pass. YuNet finds faces of
     * roughly 10-300 px, so a single scale misses a face that fills the
     * photo (close-up) or a tiny one in a large scene. */
    std::vector<int> detScales{640, 320, 160};

private:
    std::unique_ptr<Ort::Env>     env;
    std::unique_ptr<Ort::Session> detSession;
    std::unique_ptr<Ort::Session> recSession;

    std::vector<std::string> detInNames, detOutNames;
    std::vector<std::string> recInNames, recOutNames;

    int detInputSize = 640;      /* read from the model in load()          */

    /* One detector pass with the image's longest side scaled to maxSide.
     * Appends every candidate above detThreshold (no NMS) to out. */
    void detectAtScale(const uint8_t *rgb, int width, int height,
                       int maxSide, std::vector<FaceDet> &out);

    void warpToTemplate(const uint8_t *rgb, int width, int height,
                        const FaceDet &face, float *out112x112x3,
                        uint8_t *outBytes);
};

}  /* namespace essi */

#endif /* FACE_ENGINE_H_ */
