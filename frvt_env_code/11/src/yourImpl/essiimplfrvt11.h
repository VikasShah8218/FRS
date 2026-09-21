/*
 * This software was developed at the National Institute of Standards and
 * Technology (NIST) by employees of the Federal Government in the course
 * of their official duties. Pursuant to title 17 Section 105 of the
 * United States Code, this software is not subject to copyright protection
 * and is in the public domain. NIST assumes no responsibility  whatsoever for
 * its use by other parties, and makes no guarantees, expressed or implied,
 * about its quality, reliability, or any other characteristic.
 */

#ifndef ESSIIMPLFRVT11_H_
#define ESSIIMPLFRVT11_H_

#include <memory>

#include "frvt11.h"
#include "face_engine.h"

/*
 * Declare the implementation class of the FRVT 1:1 Interface
 */
namespace FRVT_11 {

class EssiImplFRVT11 : public FRVT_11::Interface {
public:

    EssiImplFRVT11();
    ~EssiImplFRVT11() override;

    FRVT::ReturnStatus
    initialize(const std::string &configDir) override;

    FRVT::ReturnStatus
    createFaceTemplate(
        const std::vector<FRVT::Image> &faces,
        FRVT::TemplateRole role,
        std::vector<uint8_t> &templ,
        std::vector<FRVT::EyePair> &eyeCoordinates) override;

    FRVT::ReturnStatus
    createIrisTemplate(
        const std::vector<FRVT::Image> &irises,
        FRVT::TemplateRole role,
        std::vector<uint8_t> &templ,
        std::vector<FRVT::IrisAnnulus> &irisLocations) override;

    FRVT::ReturnStatus
    createFaceTemplate(
        const FRVT::Image &image,
        FRVT::TemplateRole role,
        std::vector<std::vector<uint8_t>> &templs,
        std::vector<FRVT::EyePair> &eyeCoordinates) override;

    FRVT::ReturnStatus
    matchTemplates(
        const std::vector<uint8_t> &verifTemplate,
        const std::vector<uint8_t> &enrollTemplate,
        double &score) override;

    static std::shared_ptr<FRVT_11::Interface>
    getImplementation();

private:
    std::string configDir;

    /* 512 floats from ESSI-FR v1 (essi_fr_v1_last.onnx) */
    static const int featureVectorSize{512};
    static const size_t templateBytes{featureVectorSize * sizeof(float)};

    std::unique_ptr<essi::FaceEngine> engine;

    /* Convert an FRVT::Image (8-bit grey or 24-bit RGB) into an
     * interleaved RGB buffer. Returns false if the image is unusable. */
    bool toRGB(const FRVT::Image &img, std::vector<uint8_t> &rgb) const;

    /* Detect + embed the highest-scoring face in one image.
     * On success fills emb (512 floats) and eyes. Returns false if no face. */
    bool embedBest(const FRVT::Image &img,
                   std::vector<float> &emb,
                   FRVT::EyePair &eyes) const;

    /* Pack / unpack the template blob: 512 floats (2048 bytes), or ZERO
     * bytes for a failed template. pack() emits zero bytes when emb is not
     * 512 floats; unpack() returns false for anything but 2048 bytes. */
    static void pack(const std::vector<float> &emb, std::vector<uint8_t> &templ);
    static bool unpack(const std::vector<uint8_t> &templ, std::vector<float> &emb);
};

}

#endif /* ESSIIMPLFRVT11_H_ */
