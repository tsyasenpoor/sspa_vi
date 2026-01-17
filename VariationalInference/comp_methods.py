from sklearn.svm import SVC
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import NMF
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import joblib


def train_alg(
    algorithm, x_data_train, x_aux_data_train, y_data_train, save_path, latent_dim
):
    x_data_train_np = x_data_train.numpy()
    x_aux_data_train_np = x_aux_data_train.numpy()
    y_data_train_np = y_data_train.numpy().ravel()  # turn into row vector
    if algorithm == "svm":
        # SVM using gene expression and x_aux as covariates
        # Concatenate x_data_train and x_aux_data_train along the feature axis
        X_train = np.concatenate((x_data_train_np, x_aux_data_train_np), axis=1)

        # Create an SVM model with a standard scaler
        svm_model = make_pipeline(StandardScaler(), SVC(probability=True))

        # Train the SVM model
        svm_model.fit(X_train, y_data_train_np)
        train_accuracy = svm_model.score(X_train, y_data_train_np)
        print(f"Training accuracy: {train_accuracy:.2f}")

        # Optionally, save the trained model
        model_save_path = "svm_model.pkl"
        joblib.dump(svm_model, os.path.join(save_path, model_save_path))
        return svm_model
    elif algorithm == "lr":
        # logistic regression using gene expression and x_aux as covariates
        # Concatenate x_data_train and x_aux_data_train along the feature axis
        X_train = np.concatenate((x_data_train_np, x_aux_data_train_np), axis=1)

        # Create a Logistic Regression model with a standard scaler
        lr_model = make_pipeline(StandardScaler(), LogisticRegression())

        # Train the Logistic Regression model
        lr_model.fit(X_train, y_data_train_np)
        train_accuracy = lr_model.score(X_train, y_data_train_np)
        print(f"Training accuracy: {train_accuracy:.2f}")

        # Optionally, save the trained model
        model_save_path = "lr_model.pkl"
        joblib.dump(lr_model, os.path.join(save_path, model_save_path))
        return lr_model

    elif algorithm == "lrl":
        # logistic regression with lasso penalty using gene expression and x_aux as covariates
        X_train = np.concatenate((x_data_train_np, x_aux_data_train_np), axis=1)
        lrl_model = make_pipeline(
            StandardScaler(), LogisticRegression(penalty="l1", solver="saga")
        )
        lrl_model.fit(X_train, y_data_train_np)
        train_accuracy = lrl_model.score(X_train, y_data_train_np)
        print(f"Training accuracy: {train_accuracy:.2f}")
        model_save_path = "lrl_model.pkl"
        joblib.dump(lrl_model, os.path.join(save_path, model_save_path))
        return lrl_model

    elif algorithm == "lrr":
        # logistic regression with ridge penalty using gene expression and x_aux as covariates
        X_train = np.concatenate((x_data_train_np, x_aux_data_train_np), axis=1)
        lrr_model = make_pipeline(StandardScaler(), LogisticRegression(penalty="l2"))
        lrr_model.fit(X_train, y_data_train_np)
        train_accuracy = lrr_model.score(X_train, y_data_train_np)
        print(f"Training accuracy: {train_accuracy:.2f}")
        model_save_path = "lrr_model.pkl"
        joblib.dump(lrr_model, os.path.join(save_path, model_save_path))
        return lrr_model

    elif algorithm == "mflr":
        # matrix factorization on the gene expression matrix followed by latent sample factors and x_aux as covariates in an LR model
        nmf = NMF(n_components=latent_dim)
        X_latent = nmf.fit_transform(x_data_train_np)
        X_train = np.concatenate((X_latent, x_aux_data_train_np), axis=1)
        mflr_model = make_pipeline(StandardScaler(), LogisticRegression())
        mflr_model.fit(X_train, y_data_train_np)
        train_accuracy = mflr_model.score(X_train, y_data_train_np)
        print(f"Training accuracy: {train_accuracy:.2f}")
        model_save_path = "mflr_model.pkl"
        joblib.dump(mflr_model, os.path.join(save_path, model_save_path))
        return mflr_model

    elif algorithm == "mflrl":
        # matrix factorization on the gene expression matrix followed by latent sample factors and
        # x_aux as covariates in a lasso LR model
        nmf = NMF(n_components=latent_dim)
        X_latent = nmf.fit_transform(x_data_train_np)
        X_train = np.concatenate((X_latent, x_aux_data_train_np), axis=1)
        mflrl_model = make_pipeline(
            StandardScaler(), LogisticRegression(penalty="l1", solver="saga")
        )
        mflrl_model.fit(X_train, y_data_train_np)
        train_accuracy = mflrl_model.score(X_train, y_data_train_np)
        print(f"Training accuracy: {train_accuracy:.2f}")
        model_save_path = "mflrl_model.pkl"
        joblib.dump(mflrl_model, os.path.join(save_path, model_save_path))
        return mflrl_model

    elif algorithm == "mflrr":
        # matrix factorization on the gene expression matrix followed by latent sample factors and x_aux as
        # covariates in a ridge LR model
        nmf = NMF(n_components=latent_dim)
        X_latent = nmf.fit_transform(x_data_train_np)
        X_train = np.concatenate((X_latent, x_aux_data_train_np), axis=1)
        mflrr_model = make_pipeline(StandardScaler(), LogisticRegression(penalty="l2"))
        mflrr_model.fit(X_train, y_data_train_np)
        train_accuracy = mflrr_model.score(X_train, y_data_train_np)
        print(f"Training accuracy: {train_accuracy:.2f}")
        model_save_path = "mflrr_model.pkl"
        joblib.dump(mflrr_model, os.path.join(save_path, model_save_path))
        return mflrr_model

    else:
        raise ValueError("Unknown algorithm specified")


def eval_alg(
    model, algorithm, x_data_test, x_aux_data_test, y_data_test, save_path, latent_dim):
    print("evaluating " + algorithm)

    # Convert torch tensors to numpy arrays
    x_data_test_np = x_data_test.numpy()
    x_aux_data_test_np = x_aux_data_test.numpy()
    y_data_test_np = y_data_test.numpy()

    # this is a hack to check if the method is matrix factorization based... fix later
    if algorithm[0] == "m":
        nmf = NMF(n_components=latent_dim)
        X_latent = nmf.fit_transform(x_data_test_np)
        X_test = np.concatenate((X_latent, x_aux_data_test_np), axis=1)
    else:
        # Concatenate x_data_test and x_aux_data_test along the feature axis
        X_test = np.concatenate((x_data_test_np, x_aux_data_test_np), axis=1)

    # Predict the labels for the test data
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Assuming binary classification

    # Calculate the evaluation metrics
    test_accuracy = accuracy_score(y_data_test_np, y_pred)
    test_precision = precision_score(y_data_test_np, y_pred, average="weighted")
    test_recall = recall_score(y_data_test_np, y_pred, average="weighted")
    test_f1 = f1_score(y_data_test_np, y_pred, average="weighted")
    test_confusion_matrix = confusion_matrix(y_data_test_np, y_pred)
    test_roc_auc = roc_auc_score(y_data_test_np, y_pred_proba)

    print(f"Test accuracy: {test_accuracy:.2f}")
    print(f"Test precision: {test_precision:.2f}")
    print(f"Test recall: {test_recall:.2f}")
    print(f"Test F1-score: {test_f1:.2f}")
    print(f"Test ROC AUC: {test_roc_auc:.2f}")
    print(f"Confusion matrix:\n{test_confusion_matrix}")

    # Save the results to the specified save_path
    results = {
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_roc_auc": test_roc_auc,
        "confusion_matrix": test_confusion_matrix,
        "y_true": y_data_test_np,
        "y_pred": y_pred,
    }
    results_save_path = "{}_results.pkl".format(algorithm)
    joblib.dump(results, os.path.join(save_path, results_save_path))
