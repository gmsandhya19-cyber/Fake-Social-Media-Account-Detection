# Project review — 9 October 2026

## Scope
Reviewed active Flask routes, feature validation, model loading, CSV uploads and downloads, shared templates, training and evaluation, dataset structure, dependencies, and notebook code. Historical unused templates and static assets remain in place. Browser appearance was not visually inspected during this review.

## Fixed
- Detect duplicate CSV headers before pandas renames them.
- Reject values outside the Random Forest numeric range before inference.
- Share one feature schema and validator between training and prediction.
- Bound CSV parsing to the supported row limit.
- Show actionable 503 pages for missing/incompatible prediction models or unavailable metric files.
- Reload the cached prediction model after the model file changes.
- Validate binary training labels and reject contradictory labels for identical features.
- Save evaluation JSON through a temporary file; record training timestamp and dataset hash.
- Replace broken contact/empty-link controls with functional home, about and dataset pages using shared navigation.
- Replace notebook undefined variables and separate model overwrites with the shared training pipeline.

## Verification
- 12 unittest regression tests passed without warnings.
- Every notebook code cell executed in order without errors.
- pip check reported no broken requirements.
- Saved Random Forest, Decision Tree and Logistic Regression test accuracies match model_metrics.json.
- Active pages, internal navigation and linked CSS return HTTP 200 through Flask's test client.

## Results
696 original rows; 4 exact duplicates removed; 692 unique rows. Stratified split: 484 training, 208 test. Random Forest accuracy 93.75%, Decision Tree 90.87%, Logistic Regression 91.35%. Random Forest made 5 genuine-to-fake and 8 fake-to-genuine errors on the held-out split. Scores describe this split, not unseen social platforms.

## Best next additions
1. Document dataset collection, labeling criteria, consent/licensing, collection date and target platform. Add independently labeled recent accounts before claiming real-world reliability.
2. Add repeated or stratified cross-validation and an independent external evaluation set. Fit preprocessing inside each training fold.
3. Evaluate confidence calibration and a review-needed outcome for uncertain predictions. Choose thresholds on validation data, not the final test set.
4. If storing history is wanted, define retention and access requirements before adding a database and authentication.
5. Before public hosting, test the deployed server and browser layouts, and add deployment-specific operational controls. Current testing does not establish production readiness.

## Operational notes
Run app.py with the project virtual environment. If models need rebuilding, run train_model.py with that same environment. The app processes uploaded CSVs without persisting them, though the framework may spool larger multipart requests to temporary storage. No automated social profile fetching is implemented.
