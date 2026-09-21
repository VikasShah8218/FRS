/*
 * This software was developed at the National Institute of Standards and
 * Technology (NIST) by employees of the Federal Government in the course
 * of their official duties. Pursuant to title 17 Section 105 of the
 * United States Code, this software is not subject to copyright protection
 * and is in the public domain. NIST assumes no responsibility  whatsoever for
 * its use by other parties, and makes no guarantees, expressed or implied,
 * about its quality, reliability, or any other characteristic.
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iomanip>

#include "essiimplfrvt11.h"

using namespace std;
using namespace FRVT;
using namespace FRVT_11;

/* ---------------------------------------------------------------------- */

EssiImplFRVT11::EssiImplFRVT11() {}

EssiImplFRVT11::~EssiImplFRVT11() {}

/* ---------------------------------------------------------------------- */
/* Initialization                                                          */
/* ---------------------------------------------------------------------- */

ReturnStatus
EssiImplFRVT11::initialize(const std::string &configDir)
{
    this->configDir = configDir;

    this->engine.reset(new essi::FaceEngine());

    /* Models are read from configDir. NEVER hard-code a path here -
     * the validation script checks for that and will fail you. */
    const std::string err = this->engine->load(configDir);
    if (!err.empty()) {
        this->engine.reset();
        return ReturnStatus(ReturnCode::ConfigError, err);
    }
    return ReturnStatus(ReturnCode::Success);
}

/* ---------------------------------------------------------------------- */
/* Helpers                                                                 */
/* ---------------------------------------------------------------------- */

bool
EssiImplFRVT11::toRGB(const FRVT::Image &img, std::vector<uint8_t> &rgb) const
{
    if (img.data == nullptr || img.width == 0 || img.height == 0)
        return false;

    const size_t npix = static_cast<size_t>(img.width) * img.height;
    const uint8_t *src = img.data.get();
    if (src == nullptr) return false;

    rgb.resize(npix * 3);

    if (img.depth == 24) {
        /* NIST already hands us R,G,B interleaved - no channel swap needed */
        std::memcpy(rgb.data(), src, npix * 3);
    } else if (img.depth == 8) {
        /* 8-bit greyscale: replicate into all three channels */
        for (size_t i = 0; i < npix; i++) {
            rgb[i * 3 + 0] = src[i];
            rgb[i * 3 + 1] = src[i];
            rgb[i * 3 + 2] = src[i];
        }
    } else {
        return false;
    }
    return true;
}

bool
EssiImplFRVT11::embedBest(const FRVT::Image &img,
                          std::vector<float> &emb,
                          FRVT::EyePair &eyes) const
{
    emb.clear();
    if (!this->engine) return false;

    std::vector<uint8_t> rgb;
    if (!toRGB(img, rgb)) return false;

    auto faces = this->engine->detect(rgb.data(), img.width, img.height);
    if (faces.empty()) return false;

    /* detect() returns highest score first */
    const essi::FaceDet &f = faces[0];

    emb = this->engine->embed(rgb.data(), img.width, img.height, f);
    if (emb.size() != static_cast<size_t>(featureVectorSize)) {
        emb.clear();
        return false;
    }

    /* IMPORTANT eye convention.
     * YuNet kps[0] is the eye on the LEFT of the image (YuNet calls it the
     * "right eye": the subject's right), kps[1] the one on the right.
     * NIST's EyePair wants the SUBJECT's left/right - and the subject's
     * left eye appears on the RIGHT side of a frontal image.
     * So: subject-left  = kps[1],  subject-right = kps[0]. */
    auto clampX = [&](float v) -> uint16_t {
        if (v < 0) v = 0;
        if (v > img.width  - 1) v = img.width  - 1;
        return static_cast<uint16_t>(v + 0.5f);
    };
    auto clampY = [&](float v) -> uint16_t {
        if (v < 0) v = 0;
        if (v > img.height - 1) v = img.height - 1;
        return static_cast<uint16_t>(v + 0.5f);
    };

    eyes = EyePair(true, true,
                   clampX(f.kps[1][0]), clampY(f.kps[1][1]),   /* left  */
                   clampX(f.kps[0][0]), clampY(f.kps[0][1]));  /* right */
    return true;
}

void
EssiImplFRVT11::pack(const std::vector<float> &emb, std::vector<uint8_t> &templ)
{
    /* A failed template is ZERO bytes (allowed by API 4.4.3/4.4.4);
     * matchTemplates() turns it into score -1 + VerifTemplateError. */
    templ.clear();
    if (emb.size() != static_cast<size_t>(featureVectorSize))
        return;
    templ.resize(templateBytes);
    std::memcpy(templ.data(), emb.data(), templateBytes);
}

bool
EssiImplFRVT11::unpack(const std::vector<uint8_t> &templ, std::vector<float> &emb)
{
    if (templ.size() != templateBytes) return false;
    emb.resize(featureVectorSize);
    std::memcpy(emb.data(), templ.data(), templateBytes);
    return true;
}

/* ---------------------------------------------------------------------- */
/* Enrollment / verification template from one or more images of ONE person */
/* ---------------------------------------------------------------------- */

ReturnStatus
EssiImplFRVT11::createFaceTemplate(
    const std::vector<FRVT::Image> &faces,
    TemplateRole role,
    std::vector<uint8_t> &templ,
    std::vector<EyePair> &eyeCoordinates)
{
    if (!this->engine)
        return ReturnStatus(ReturnCode::ConfigError, "not initialized");

    std::vector<float> sum(featureVectorSize, 0.0f);
    int nGood = 0;

    /* One EyePair MUST be pushed for every input image, in order. */
    for (const auto &img : faces) {
        std::vector<float> emb;
        EyePair eyes;

        if (embedBest(img, emb, eyes)) {
            for (int i = 0; i < featureVectorSize; i++)
                sum[i] += emb[i];
            nGood++;
            eyeCoordinates.push_back(eyes);
        } else {
            /* No face found - still push a placeholder so the counts line up */
            eyeCoordinates.push_back(EyePair(false, false, 0, 0, 0, 0));
        }
    }

    if (nGood == 0) {
        /* Failed template: zero bytes. matchTemplates() accepts it and
         * returns score -1 + VerifTemplateError, as API 4.4.5 requires. */
        templ.clear();
        return ReturnStatus(ReturnCode::FaceDetectionError,
                            "no face detected in any input image");
    }

    /* Average the normalized embeddings, then re-normalize.
     * Standard practice for multi-image enrollment. */
    double norm = 0.0;
    for (int i = 0; i < featureVectorSize; i++) {
        sum[i] /= static_cast<float>(nGood);
        norm += static_cast<double>(sum[i]) * sum[i];
    }
    norm = std::sqrt(norm);
    if (norm > 1e-9)
        for (int i = 0; i < featureVectorSize; i++)
            sum[i] = static_cast<float>(sum[i] / norm);

    pack(sum, templ);
    return ReturnStatus(ReturnCode::Success);
}

/* ---------------------------------------------------------------------- */
/* Iris - not accepted for 1:1 right now. Leave as-is.                      */
/* ---------------------------------------------------------------------- */

ReturnStatus
EssiImplFRVT11::createIrisTemplate(
    const std::vector<FRVT::Image> &irises,
    TemplateRole role,
    std::vector<uint8_t> &templ,
    std::vector<IrisAnnulus> &irisLocations)
{
    return ReturnStatus(ReturnCode::NotImplemented);
}

/* ---------------------------------------------------------------------- */
/* One image that may contain SEVERAL different people                      */
/* ---------------------------------------------------------------------- */

ReturnStatus
EssiImplFRVT11::createFaceTemplate(
    const FRVT::Image &image,
    FRVT::TemplateRole role,
    std::vector<std::vector<uint8_t>> &templs,
    std::vector<FRVT::EyePair> &eyeCoordinates)
{
    if (!this->engine)
        return ReturnStatus(ReturnCode::ConfigError, "not initialized");

    /* Zero faces: exactly one template, zero bytes (failed), plus a
     * non-successful return code (API 4.4.4, Table 6). */
    std::vector<uint8_t> rgb;
    if (!toRGB(image, rgb)) {
        templs.push_back(std::vector<uint8_t>());
        eyeCoordinates.push_back(EyePair(false, false, 0, 0, 0, 0));
        return ReturnStatus(ReturnCode::FaceDetectionError, "bad image");
    }

    auto faces = this->engine->detect(rgb.data(), image.width, image.height);

    if (faces.empty()) {
        templs.push_back(std::vector<uint8_t>());
        eyeCoordinates.push_back(EyePair(false, false, 0, 0, 0, 0));
        return ReturnStatus(ReturnCode::FaceDetectionError, "no face detected");
    }

    /* Cap the number of faces so a crowd scene cannot blow the time limit. */
    const size_t maxFaces = 10;
    if (faces.size() > maxFaces) faces.resize(maxFaces);

    auto clampX = [&](float v) -> uint16_t {
        if (v < 0) v = 0;
        if (v > image.width  - 1) v = image.width  - 1;
        return static_cast<uint16_t>(v + 0.5f);
    };
    auto clampY = [&](float v) -> uint16_t {
        if (v < 0) v = 0;
        if (v > image.height - 1) v = image.height - 1;
        return static_cast<uint16_t>(v + 0.5f);
    };

    for (const auto &f : faces) {
        std::vector<float> emb =
            this->engine->embed(rgb.data(), image.width, image.height, f);

        std::vector<uint8_t> t;
        pack(emb, t);            /* zero bytes (failed) if embedding failed */
        templs.push_back(t);

        /* subject-left = kps[1], subject-right = kps[0] */
        eyeCoordinates.push_back(
            EyePair(true, true,
                    clampX(f.kps[1][0]), clampY(f.kps[1][1]),
                    clampX(f.kps[0][0]), clampY(f.kps[0][1])));
    }

    return ReturnStatus(ReturnCode::Success);
}

/* ---------------------------------------------------------------------- */
/* Matching                                                                 */
/* ---------------------------------------------------------------------- */

ReturnStatus
EssiImplFRVT11::matchTemplates(
    const std::vector<uint8_t> &verifTemplate,
    const std::vector<uint8_t> &enrollTemplate,
    double &score)
{
    /* API 4.4.5: when either template comes from a failed template
     * generation (zero bytes here) the score SHALL be -1 and the return
     * value VerifTemplateError. A template of any other wrong size is
     * treated the same way. */
    score = -1.0;

    std::vector<float> a, b;
    if (!unpack(verifTemplate, a) || !unpack(enrollTemplate, b))
        return ReturnStatus(ReturnCode::VerifTemplateError,
                            "failed or malformed template");

    double dot = 0.0;
    for (int i = 0; i < featureVectorSize; i++)
        dot += static_cast<double>(a[i]) * b[i];

    if (dot >  1.0) dot =  1.0;
    if (dot < -1.0) dot = -1.0;

    /* Cosine is -1..1 but scores of successful matches must be
     * non-negative (API 4.4.5). Map linearly to 0..100 - do not
     * clamp at zero, that would throw away real information. */
    score = (dot + 1.0) * 50.0;

    return ReturnStatus(ReturnCode::Success);
}

/* ---------------------------------------------------------------------- */

std::shared_ptr<Interface>
Interface::getImplementation()
{
    return std::make_shared<EssiImplFRVT11>();
}
