# Data

`images/` holds 7 sample images bundled with scikit-image (astronaut, cat/chelsea, coffee, rocket, cameraman) and scikit-learn (china, flower). Both libraries ship them for demonstration; scikit-learn's two photos are Creative Commons licensed, so check the scikit-learn dataset documentation for attribution before redistributing.

`labels.json` is the single source of truth: image names, the 12 core texts (T0-T11), valid matches (astronaut has two: T0 and T7), and 8 `hard_distractors` that have no valid image and are used only in the separately reported expanded run.
