use crate::cancer;
use crate::impl_subclass;
use rand::rng;

trait Subclass {
    fn into_parts(self: Box<Self>) -> (Vec<Vec<Vec<Vec<f64>>>>, (Vec<Vec<f64>>, Vec<Vec<f64>>));
}

impl_subclass!(
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
);

pub struct Dataset(
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

const slice: usize = 100;

impl Dataset {
    //use if need all images
    pub fn flatten_into_4d(self) -> (Vec<Vec<Vec<Vec<f64>>>>, (Vec<Vec<f64>>, Vec<Vec<f64>>)) {
        let mut image_storage = Vec::new();
        let mut class_label_storage = Vec::new();
        let mut subclass_label_storage = Vec::new();

        // Acute Lymphoblastic Leukemia
        Self::append(
            Box::new(self.0.0),
            &mut image_storage,
            &mut class_label_storage,
            &mut subclass_label_storage,
        );
        Self::append(
            Box::new(self.0.1),
            &mut image_storage,
            &mut class_label_storage,
            &mut subclass_label_storage,
        );
        Self::append(
            Box::new(self.0.2),
            &mut image_storage,
            &mut class_label_storage,
            &mut subclass_label_storage,
        );
        Self::append(
            Box::new(self.0.3),
            &mut image_storage,
            &mut class_label_storage,
            &mut subclass_label_storage,
        );

        // Brain Cancer
        Self::append(
            Box::new(self.1.0),
            &mut image_storage,
            &mut class_label_storage,
            &mut subclass_label_storage,
        );
        Self::append(
            Box::new(self.1.1),
            &mut image_storage,
            &mut class_label_storage,
            &mut subclass_label_storage,
        );
        Self::append(
            Box::new(self.1.2),
            &mut image_storage,
            &mut class_label_storage,
            &mut subclass_label_storage,
        );

        // Breast Cancer
        Self::append(
            Box::new(self.2.0),
            &mut image_storage,
            &mut class_label_storage,
            &mut subclass_label_storage,
        );
        Self::append(
            Box::new(self.2.1),
            &mut image_storage,
            &mut class_label_storage,
            &mut subclass_label_storage,
        );

        // Cervical Cancer
        Self::append(
            Box::new(self.3.0),
            &mut image_storage,
            &mut class_label_storage,
            &mut subclass_label_storage,
        );
        Self::append(
            Box::new(self.3.1),
            &mut image_storage,
            &mut class_label_storage,
            &mut subclass_label_storage,
        );
        Self::append(
            Box::new(self.3.2),
            &mut image_storage,
            &mut class_label_storage,
            &mut subclass_label_storage,
        );
        Self::append(
            Box::new(self.3.3),
            &mut image_storage,
            &mut class_label_storage,
            &mut subclass_label_storage,
        );
        Self::append(
            Box::new(self.3.4),
            &mut image_storage,
            &mut class_label_storage,
            &mut subclass_label_storage,
        );

        // Kidney Cancer
        Self::append(
            Box::new(self.4.0),
            &mut image_storage,
            &mut class_label_storage,
            &mut subclass_label_storage,
        );
        Self::append(
            Box::new(self.4.1),
            &mut image_storage,
            &mut class_label_storage,
            &mut subclass_label_storage,
        );

        // Lung Cancer
        Self::append(
            Box::new(self.5.0),
            &mut image_storage,
            &mut class_label_storage,
            &mut subclass_label_storage,
        );
        Self::append(
            Box::new(self.5.1),
            &mut image_storage,
            &mut class_label_storage,
            &mut subclass_label_storage,
        );
        Self::append(
            Box::new(self.5.2),
            &mut image_storage,
            &mut class_label_storage,
            &mut subclass_label_storage,
        );

        // Colon Cancer
        Self::append(
            Box::new(self.6.0),
            &mut image_storage,
            &mut class_label_storage,
            &mut subclass_label_storage,
        );
        Self::append(
            Box::new(self.6.1),
            &mut image_storage,
            &mut class_label_storage,
            &mut subclass_label_storage,
        );

        // Lymphoma
        Self::append(
            Box::new(self.7.0),
            &mut image_storage,
            &mut class_label_storage,
            &mut subclass_label_storage,
        );
        Self::append(
            Box::new(self.7.1),
            &mut image_storage,
            &mut class_label_storage,
            &mut subclass_label_storage,
        );
        Self::append(
            Box::new(self.7.2),
            &mut image_storage,
            &mut class_label_storage,
            &mut subclass_label_storage,
        );

        // Oral Cancer
        Self::append(
            Box::new(self.8.0),
            &mut image_storage,
            &mut class_label_storage,
            &mut subclass_label_storage,
        );
        Self::append(
            Box::new(self.8.1),
            &mut image_storage,
            &mut class_label_storage,
            &mut subclass_label_storage,
        );

        (image_storage, (class_label_storage, subclass_label_storage))
    }

    fn append(
        d: Box<dyn Subclass>,
        image_storage: &mut Vec<Vec<Vec<Vec<f64>>>>,
        class_label_storage: &mut Vec<Vec<f64>>,
        subclass_label_storage: &mut Vec<Vec<f64>>,
    ) {
        let (images, (class, sub)) = d.into_parts();

        image_storage.extend(images);
        class_label_storage.extend(class);
        subclass_label_storage.extend(sub);
    }

    fn shuffle() {}
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
pub enum Subclasses {
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

pub fn new() -> Dataset {
    // Acute Lymphoblastic Leukemia
    let acute_lymphoblastic_leukemia_benign: AcuteLymphoblasticLeukemiaBenign =
        AcuteLymphoblasticLeukemiaBenign(
            cancer::load_dataset("datasets/multi_cancer/Multi Cancer/ALL/all_benign", &slice)
                .expect("Unwrap image error"),
            cancer::generate_labels(
                &slice,
                Class::AcuteLymphoblasticLeukemia,
                Subclasses::AcuteLymphoblasticLeukemiaBenign,
            ),
        );

    let acute_lymphoblastic_leukemia_early: AcuteLymphoblasticLeukemiaEarly =
        AcuteLymphoblasticLeukemiaEarly(
            cancer::load_dataset("datasets/multi_cancer/Multi Cancer/ALL/all_early", &slice)
                .expect("Unwrap image error"),
            cancer::generate_labels(
                &slice,
                Class::AcuteLymphoblasticLeukemia,
                Subclasses::AcuteLymphoblasticLeukemiaEarly,
            ),
        );

    let acute_lymphoblastic_leukemia_pre: AcuteLymphoblasticLeukemiaPre =
        AcuteLymphoblasticLeukemiaPre(
            cancer::load_dataset("datasets/multi_cancer/Multi Cancer/ALL/all_pre", &slice)
                .expect("Unwrap image error"),
            cancer::generate_labels(
                &slice,
                Class::AcuteLymphoblasticLeukemia,
                Subclasses::AcuteLymphoblasticLeukemiaPre,
            ),
        );

    let acute_lymphoblastic_leukemia_pro: AcuteLymphoblasticLeukemiaPro =
        AcuteLymphoblasticLeukemiaPro(
            cancer::load_dataset("datasets/multi_cancer/Multi Cancer/ALL/all_pro", &slice)
                .expect("Unwrap image error"),
            cancer::generate_labels(
                &slice,
                Class::AcuteLymphoblasticLeukemia,
                Subclasses::AcuteLymphoblasticLeukemiaPro,
            ),
        );

    // Brain Cancer
    let brain_glioma: BrainGlioma = BrainGlioma(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Brain Cancer/brain_glioma",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(&slice, Class::BrainCancer, Subclasses::BrainGlioma),
    );

    let brain_meningioma: BrainMeningioma = BrainMeningioma(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Brain Cancer/brain_menin",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(&slice, Class::BrainCancer, Subclasses::BrainMeningioma),
    );

    let brain_tumor: BrainTumor = BrainTumor(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Brain Cancer/brain_tumor",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(&slice, Class::BrainCancer, Subclasses::BrainTumor),
    );

    // Breast Cancer
    let breast_benign: BreastBenign = BreastBenign(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Breast Cancer/breast_benign",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(&slice, Class::BreastCancer, Subclasses::BreastBenign),
    );

    let breast_malignant: BreastMalignant = BreastMalignant(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Breast Cancer/breast_malignant",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(&slice, Class::BreastCancer, Subclasses::BreastMalignant),
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
            Subclasses::CervicalDyskeratotic,
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
            Subclasses::CervicalKoilocytotic,
        ),
    );

    let cervical_metaplastic: CervicalMetaplastic = CervicalMetaplastic(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Cervical Cancer/cervix_mep",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(
            &slice,
            Class::CervicalCancer,
            Subclasses::CervicalMetaplastic,
        ),
    );

    let cervical_parabasal: CervicalParabasal = CervicalParabasal(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Cervical Cancer/cervix_pab",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(&slice, Class::CervicalCancer, Subclasses::CervicalParabasal),
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
                Subclasses::CervicalSuperficialIntermediate,
            ),
        );

    // Kidney Cancer
    let kidney_normal: KidneyNormal = KidneyNormal(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Kidney Cancer/kidney_normal",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(&slice, Class::KidneyCancer, Subclasses::KidneyNormal),
    );

    let kidney_tumor: KidneyTumor = KidneyTumor(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Kidney Cancer/kidney_tumor",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(&slice, Class::KidneyCancer, Subclasses::KidneyTumor),
    );

    // Lung and Colon Cancer
    let colon_adenocarcinoma: ColonAdenocarcinoma = ColonAdenocarcinoma(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Lung and Colon Cancer/colon_aca",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(&slice, Class::ColonCancer, Subclasses::ColonAdenocarcinoma),
    );

    let colon_benign: ColonBenign = ColonBenign(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Lung and Colon Cancer/colon_bnt",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(&slice, Class::ColonCancer, Subclasses::ColonBenign),
    );

    let lung_adenocarcinoma: LungAdenocarcinoma = LungAdenocarcinoma(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Lung and Colon Cancer/lung_aca",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(&slice, Class::LungCancer, Subclasses::LungAdenocarcinoma),
    );

    let lung_benign: LungBenign = LungBenign(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Lung and Colon Cancer/lung_bnt",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(&slice, Class::LungCancer, Subclasses::LungBenign),
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
            Subclasses::LungSquamousCellCarcinoma,
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
                Subclasses::LymphomaChronicLymphocyticLeukemia,
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
            Subclasses::LymphomaFollicularLymphoma,
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
            Subclasses::LymphomaMantleCellLymphoma,
        ),
    );

    // Oral Cancer
    let oral_normal: OralNormal = OralNormal(
        cancer::load_dataset(
            "datasets/multi_cancer/Multi Cancer/Oral Cancer/oral_normal",
            &slice,
        )
        .expect("Unwrap image error"),
        cancer::generate_labels(&slice, Class::OralCancer, Subclasses::OralNormal),
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
            Subclasses::OralOralSquamousCellCarcinoma,
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

    Dataset(
        acute_lymphoblastic_leukemia,
        brain_cancer,
        breast_cancer,
        cervical_cancer,
        kidney_cancer,
        lung_cancer,
        colon_cancer,
        lymphoma,
        oral_cancer,
    )
}
