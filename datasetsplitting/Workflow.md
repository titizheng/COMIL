1. Load patient–slide–label mapping table from a CSV file.

    Columns: patient_id, pathology_id, subtype, label.

2. Check label consistency within the same patient.

    If a patient has slides with different labels, print a warning.

3. Deduplicate to patient-level.

    Keep one row per patient (unique patient_id with its label).

4. Patient-level stratified 5-fold split.

    Use StratifiedKFold to ensure balanced class distribution across folds.

    For each fold, split patients into train set and test set.

5. Further split training patients into train/val.

    Use StratifiedShuffleSplit to create 80% train, 20% validation.

6. Save split results as .npz files containing:

    train_patients, val_patients, test_patients.

7. Map patient-level splits back to slide-level.

    Read .npz file.

    For each patient, retrieve all corresponding slides and labels.

    Save the train/val/test split into a CSV file for later training.