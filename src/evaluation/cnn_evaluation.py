import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import logging

def evaluate_cnn(model, X_test, y_test):
    """
    Evaluating CNN model.
    :param model: Trained CNN model
    :param X_test: Test data features
    :param y_test: Test data labels
    :return: Dictionary of evaluation metrics
    """
    logger = logging.getLogger('evaluation_logger')
    
    try:
        # Predicting test data
        logger.info("Predicting test data with CNN model...")
        y_pred_probs = model.predict(X_test)
        y_pred_classes = y_pred_probs.argmax(axis=1)
        y_true_classes = y_test.argmax(axis=1)
        
        # Calculating evaluation metrics
        logger.info("Calculating evaluation metrics...")
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
        precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
        recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
        report = classification_report(y_true_classes, y_pred_classes)
        conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
        
        # To log the evaluation results
        logger.info(f"CNN Model Accuracy: {accuracy}")
        logger.info(f"CNN Model F1 Score: {f1}")
        logger.info(f"CNN Model Precision: {precision}")
        logger.info(f"CNN Model Recall: {recall}")
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
        logger.error(f"Error during CNN model evaluation: {e}", exc_info=True)
        raise
