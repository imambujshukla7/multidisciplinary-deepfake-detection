from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import logging

def evaluate_bayesian(model, X_test, y_test):
    """
    Evaluating Bayesian model.
    :param model: Trained Bayesian model
    :param X_test: Test data features
    :param y_test: Test data labels
    :return: Dictionary of evaluation metrics
    """
    logger = logging.getLogger('evaluation_logger')
    
    try:
        logger.info("Evaluating Bayesian model...")
        
        # Predicting test data
        logger.info("Predicting test data with Bayesian model...")
        y_pred = model.predict(X_test)
        
        # Calculating evaluation metrics
        logger.info("Calculating evaluation metrics...")
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # To log evaluation results
        logger.info(f"Bayesian Model Accuracy: {accuracy}")
        logger.info(f"Bayesian Model F1 Score: {f1}")
        logger.info(f"Bayesian Model Precision: {precision}")
        logger.info(f"Bayesian Model Recall: {recall}")
        logger.info(f"Classification Report:\n{report}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        
        # Returning evaluation metrics
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
    
    except Exception as e:
        logger.error(f"Error during Bayesian model evaluation: {e}", exc_info=True)
        raise
