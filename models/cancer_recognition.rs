use crate::cancer;

struct Dataset(
    AcuteLymphoblasticLeukemia,
    BrainCancer,
    BreastCancer,
    CervicalCancer,
    KidneyCancer,
    LungCancer,
    ColonCancer,
    Lymphoma,
    OralCancer,
);

impl Dataset {
    fn shuffle(&mut self) {}
}

#[non_exhaustive]
pub enum Class {
    AcuteLymphoblasticLeukemia,
    BrainCancer,
    BreastCancer,
    CervicalCancer,
    KidneyCancer,
    LungCancer,
    ColonCancer,
    Lymphoma,
    OralCancer,
}

#[non_exhaustive]
pub enum Subclass {
    AcuteLymphoblasticLeukemiaBenign,
    AcuteLymphoblasticLeukemiaEarly,
    AcuteLymphoblasticLeukemiaPre,
    AcuteLymphoblasticLeukemiaPro,
    BrainGlioma,
    BrainMeningioma,
    BrainTumor,
    BreastBenign,
    BreastMalignant,
    CervicalDyskeratotic,
    CervicalKoilocytotic,
    CervicalMetaplastic,
    CervicalParabasal,
    CervicalSuperficialIntermediate,
    KidneyNormal,
    KidneyTumor,
    LungAdenocarcinoma,
    LungBenign,
    LungSquamousCellCarcinoma,
    ColonAdenocarcinoma,
    ColonBenign,
    LymphomaChronicLymphocyticLeukemia,
    LymphomaFollicularLymphoma,
    LymphomaMantleCellLymphoma,
    OralNormal,
    OralOralSquamousCellCarcinoma,
}

struct AcuteLymphoblasticLeukemia(
    AcuteLymphoblasticLeukemiaBenign,
    AcuteLymphoblasticLeukemiaEarly,
    AcuteLymphoblasticLeukemiaPre,
    AcuteLymphoblasticLeukemiaPro,
);

struct AcuteLymphoblasticLeukemiaBenign(Vec<Vec<Vec<Vec<f64>>>>, (Vec<Vec<f64>>, Vec<Vec<f64>>));
struct AcuteLymphoblasticLeukemiaEarly(Vec<Vec<Vec<Vec<f64>>>>, (Vec<Vec<f64>>, Vec<Vec<f64>>));
struct AcuteLymphoblasticLeukemiaPre(Vec<Vec<Vec<Vec<f64>>>>, (Vec<Vec<f64>>, Vec<Vec<f64>>));
struct AcuteLymphoblasticLeukemiaPro(Vec<Vec<Vec<Vec<f64>>>>, (Vec<Vec<f64>>, Vec<Vec<f64>>));

struct BrainCancer(BrainGlioma, BrainMeningioma, BrainTumor);

struct BrainGlioma(Vec<Vec<Vec<Vec<f64>>>>, (Vec<Vec<f64>>, Vec<Vec<f64>>));
struct BrainMeningioma(Vec<Vec<Vec<Vec<f64>>>>, (Vec<Vec<f64>>, Vec<Vec<f64>>));
struct BrainTumor(Vec<Vec<Vec<Vec<f64>>>>, (Vec<Vec<f64>>, Vec<Vec<f64>>));

struct BreastCancer(BreastBenign, BreastMalignant);

struct BreastBenign(Vec<Vec<Vec<Vec<f64>>>>, (Vec<Vec<f64>>, Vec<Vec<f64>>));
struct BreastMalignant(Vec<Vec<Vec<Vec<f64>>>>, (Vec<Vec<f64>>, Vec<Vec<f64>>));

struct CervicalCancer(
    CervicalDyskeratotic,
    CervicalKoilocytotic,
    CervicalMetaplastic,
    CervicalParabasal,
    CervicalSuperficialIntermediate,
);

struct CervicalDyskeratotic(Vec<Vec<Vec<Vec<f64>>>>, (Vec<Vec<f64>>, Vec<Vec<f64>>));
struct CervicalKoilocytotic(Vec<Vec<Vec<Vec<f64>>>>, (Vec<Vec<f64>>, Vec<Vec<f64>>));
struct CervicalMetaplastic(Vec<Vec<Vec<Vec<f64>>>>, (Vec<Vec<f64>>, Vec<Vec<f64>>));
struct CervicalParabasal(Vec<Vec<Vec<Vec<f64>>>>, (Vec<Vec<f64>>, Vec<Vec<f64>>));
struct CervicalSuperficialIntermediate(Vec<Vec<Vec<Vec<f64>>>>, (Vec<Vec<f64>>, Vec<Vec<f64>>));

struct KidneyCancer(KidneyNormal, KidneyTumor);

struct KidneyNormal(Vec<Vec<Vec<Vec<f64>>>>, (Vec<Vec<f64>>, Vec<Vec<f64>>));
struct KidneyTumor(Vec<Vec<Vec<Vec<f64>>>>, (Vec<Vec<f64>>, Vec<Vec<f64>>));

struct LungCancer(LungAdenocarcinoma, LungBenign, LungSquamousCellCarcinoma);

struct LungAdenocarcinoma(Vec<Vec<Vec<Vec<f64>>>>, (Vec<Vec<f64>>, Vec<Vec<f64>>));
struct LungBenign(Vec<Vec<Vec<Vec<f64>>>>, (Vec<Vec<f64>>, Vec<Vec<f64>>));
struct LungSquamousCellCarcinoma(Vec<Vec<Vec<Vec<f64>>>>, (Vec<Vec<f64>>, Vec<Vec<f64>>));

struct ColonCancer(ColonAdenocarcinoma, ColonBenign);

struct ColonAdenocarcinoma(Vec<Vec<Vec<Vec<f64>>>>, (Vec<Vec<f64>>, Vec<Vec<f64>>));
struct ColonBenign(Vec<Vec<Vec<Vec<f64>>>>, (Vec<Vec<f64>>, Vec<Vec<f64>>));

struct Lymphoma(
    LymphomaChronicLymphocyticLeukemia,
    LymphomaFollicularLymphoma,
    LymphomaMantleCellLymphoma,
);

struct LymphomaChronicLymphocyticLeukemia(Vec<Vec<Vec<Vec<f64>>>>, (Vec<Vec<f64>>, Vec<Vec<f64>>));
struct LymphomaFollicularLymphoma(Vec<Vec<Vec<Vec<f64>>>>, (Vec<Vec<f64>>, Vec<Vec<f64>>));
struct LymphomaMantleCellLymphoma(Vec<Vec<Vec<Vec<f64>>>>, (Vec<Vec<f64>>, Vec<Vec<f64>>));

struct OralCancer(OralNormal, OralOralSquamousCellCarcinoma);

struct OralNormal(Vec<Vec<Vec<Vec<f64>>>>, (Vec<Vec<f64>>, Vec<Vec<f64>>));
struct OralOralSquamousCellCarcinoma(Vec<Vec<Vec<Vec<f64>>>>, (Vec<Vec<f64>>, Vec<Vec<f64>>));

pub fn new() {
    let slice = 100;

    // Acute Lymphoblastic Leukemia
    let acute_lymphoblastic_leukemia_benign: AcuteLymphoblasticLeukemiaBenign =
        AcuteLymphoblasticLeukemiaBenign(
            cancer::load_dataset("datasets/multi_cancer/Multi Cancer/ALL/all_benign", &slice)
                .expect("Unwrap image error"),
            cancer::generate_labels(
                &slice,
                Class::AcuteLymphoblasticLeukemia,
                Subclass::AcuteLymphoblasticLeukemiaBenign,
            ),
        );

    let acute_lymphoblastic_leukemia_early: AcuteLymphoblasticLeukemiaEarly =
        AcuteLymphoblasticLeukemiaEarly(
            cancer::load_dataset("datasets/multi_cancer/Multi Cancer/ALL/all_early", &slice)
                .expect("Unwrap image error"),
            cancer::generate_labels(
                &slice,
                Class::AcuteLymphoblasticLeukemia,
                Subclass::AcuteLymphoblasticLeukemiaEarly,
            ),
        );

    let acute_lymphoblastic_leukemia_pre: AcuteLymphoblasticLeukemiaPre =
        AcuteLymphoblasticLeukemiaPre(
            cancer::load_dataset("datasets/multi_cancer/Multi Cancer/ALL/all_pre", &slice)
                .expect("Unwrap image error"),
            cancer::generate_labels(
                &slice,
                Class::AcuteLymphoblasticLeukemia,
                Subclass::AcuteLymphoblasticLeukemiaPre,
            ),
        );

    let acute_lymphoblastic_leukemia_pro: AcuteLymphoblasticLeukemiaPro =
        AcuteLymphoblasticLeukemiaPro(
            cancer::load_dataset("datasets/multi_cancer/Multi Cancer/ALL/all_pro", &slice)
                .expect("Unwrap image error"),
            cancer::generate_labels(
                &slice,
                Class::AcuteLymphoblasticLeukemia,
                Subclass::AcuteLymphoblasticLeukemiaPro,
            ),
        );

    // Brain Cancer
    let brain_glioma: BrainGlioma = BrainGlioma(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Brain Cancer/brain_glioma",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(&slice, Class::BrainCancer, Subclass::BrainGlioma),
    );

    let brain_meningioma: BrainMeningioma = BrainMeningioma(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Brain Cancer/brain_menin",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(&slice, Class::BrainCancer, Subclass::BrainMeningioma),
    );

    let brain_tumor: BrainTumor = BrainTumor(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Brain Cancer/brain_tumor",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(&slice, Class::BrainCancer, Subclass::BrainTumor),
    );

    // Breast Cancer
    let breast_benign: BreastBenign = BreastBenign(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Breast Cancer/breast_benign",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(&slice, Class::BreastCancer, Subclass::BreastBenign),
    );

    let breast_malignant: BreastMalignant = BreastMalignant(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Breast Cancer/breast_malignant",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(&slice, Class::BreastCancer, Subclass::BreastMalignant),
    );

    // Cervical Cancer
    let cervical_Dyskeratotic: CervicalDyskeratotic = CervicalDyskeratotic(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Cervical Cancer/cervix_dyk",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(
            &slice,
            Class::CervicalCancer,
            Subclass::CervicalDyskeratotic,
        ),
    );

    let cervical_koilocytotic: CervicalKoilocytotic = CervicalKoilocytotic(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Cervical Cancer/cervix_koc",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(
            &slice,
            Class::CervicalCancer,
            Subclass::CervicalKoilocytotic,
        ),
    );

    let cervical_metaplastic: CervicalMetaplastic = CervicalMetaplastic(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Cervical Cancer/cervix_mep",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(&slice, Class::CervicalCancer, Subclass::CervicalMetaplastic),
    );

    let cervical_parabasal: CervicalParabasal = CervicalParabasal(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Cervical Cancer/cervix_pab",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(&slice, Class::CervicalCancer, Subclass::CervicalParabasal),
    );

    let cervical_superficial_intermediate: CervicalSuperficialIntermediate =
        CervicalSuperficialIntermediate(
            cancer::load_dataset(
                "datasets/multi_cancer/Multi Cancer/Cervical Cancer/cervix_sfi",
                &slice,
            )
            .expect("Unwrap image error"),
            cancer::generate_labels(
                &slice,
                Class::CervicalCancer,
                Subclass::CervicalSuperficialIntermediate,
            ),
        );

    // Kidney Cancer
    let kidney_normal: KidneyNormal = KidneyNormal(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Kidney Cancer/kidney_normal",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(&slice, Class::KidneyCancer, Subclass::KidneyNormal),
    );

    let kidney_tumor: KidneyTumor = KidneyTumor(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Kidney Cancer/kidney_tumor",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(&slice, Class::KidneyCancer, Subclass::KidneyTumor),
    );

    // Lung and Colon Cancer
    let colon_adenocarcinoma: ColonAdenocarcinoma = ColonAdenocarcinoma(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Lung and Colon Cancer/colon_aca",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(&slice, Class::ColonCancer, Subclass::ColonAdenocarcinoma),
    );

    let colon_benign: ColonBenign = ColonBenign(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Lung and Colon Cancer/colon_bnt",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(&slice, Class::ColonCancer, Subclass::ColonBenign),
    );

    let lung_adenocarcinoma: LungAdenocarcinoma = LungAdenocarcinoma(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Lung and Colon Cancer/lung_aca",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(&slice, Class::LungCancer, Subclass::LungAdenocarcinoma),
    );

    let lung_benign: LungBenign = LungBenign(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Lung and Colon Cancer/lung_bnt",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(&slice, Class::LungCancer, Subclass::LungBenign),
    );

    let lung_squamous_cell_carcinoma: LungSquamousCellCarcinoma = LungSquamousCellCarcinoma(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Lung and Colon Cancer/lung_scc",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(
            &slice,
            Class::LungCancer,
            Subclass::LungSquamousCellCarcinoma,
        ),
    );

    // Lymphoma
    let lymphoma_chronic_lymphocytic_leukemia: LymphomaChronicLymphocyticLeukemia =
        LymphomaChronicLymphocyticLeukemia(
            cancer::load_dataset(
                "datasets/multi_cancer/Multi Cancer/Lymphoma/lymph_cll",
                &slice,
            )
            .expect("Unwrap image error"),
            cancer::generate_labels(
                &slice,
                Class::Lymphoma,
                Subclass::LymphomaChronicLymphocyticLeukemia,
            ),
        );

    let lymphoma_follicular_lymphoma: LymphomaFollicularLymphoma = LymphomaFollicularLymphoma(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Lymphoma/lymph_fl",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(
            &slice,
            Class::Lymphoma,
            Subclass::LymphomaFollicularLymphoma,
        ),
    );

    let lymphoma_mantle_cell_lymphoma: LymphomaMantleCellLymphoma = LymphomaMantleCellLymphoma(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Lymphoma/lymph_mcl",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(
            &slice,
            Class::Lymphoma,
            Subclass::LymphomaMantleCellLymphoma,
        ),
    );

    // Oral Cancer
    let oral_normal: OralNormal = OralNormal(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Oral Cancer/oral_normal",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(&slice, Class::OralCancer, Subclass::OralNormal),
    );

    let oral_squamous_cell_carcinoma: OralOralSquamousCellCarcinoma = OralOralSquamousCellCarcinoma(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Oral Cancer/oral_scc",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(
            &slice,
            Class::OralCancer,
            Subclass::OralOralSquamousCellCarcinoma,
        ),
    );

    let acute_lymphoblastic_leukemia: AcuteLymphoblasticLeukemia = AcuteLymphoblasticLeukemia(
        acute_lymphoblastic_leukemia_benign,
        acute_lymphoblastic_leukemia_early,
        acute_lymphoblastic_leukemia_pre,
        acute_lymphoblastic_leukemia_pro,
    );

    let brain_cancer: BrainCancer = BrainCancer(brain_glioma, brain_meningioma, brain_tumor);

    let breast_cancer: BreastCancer = BreastCancer(breast_benign, breast_malignant);

    let cervical_cancer: CervicalCancer = CervicalCancer(
        cervical_Dyskeratotic,
        cervical_koilocytotic,
        cervical_metaplastic,
        cervical_parabasal,
        cervical_superficial_intermediate,
    );

    let kidney_cancer: KidneyCancer = KidneyCancer(kidney_normal, kidney_tumor);

    let lung_cancer: LungCancer = LungCancer(
        lung_adenocarcinoma,
        lung_benign,
        lung_squamous_cell_carcinoma,
    );

    let colon_cancer: ColonCancer = ColonCancer(colon_adenocarcinoma, colon_benign);

    let lymphoma: Lymphoma = Lymphoma(
        lymphoma_chronic_lymphocytic_leukemia,
        lymphoma_follicular_lymphoma,
        lymphoma_mantle_cell_lymphoma,
    );

    let oral_cancer: OralCancer = OralCancer(oral_normal, oral_squamous_cell_carcinoma);

    //begins unshuffled
    let dataset: Dataset = Dataset(
        acute_lymphoblastic_leukemia,
        brain_cancer,
        breast_cancer,
        cervical_cancer,
        kidney_cancer,
        lung_cancer,
        colon_cancer,
        lymphoma,
        oral_cancer,
    );

    //dataset.shuffle();
}
