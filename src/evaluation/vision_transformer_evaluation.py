import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import logging

def evaluate_vision_transformer(model, dataloader, device):
    """
    Evaluating Vision Transformer model.
    :param model: Trained Vision Transformer model
    :param dataloader: DataLoader for the test data
    :param device: Device to perform evaluation on ('cpu' or 'cuda')
    :return: Dictionary of evaluation metrics
    """
    logger = logging.getLogger('evaluation_logger')
    
    try:
        logger.info("Evaluating Vision Transformer model...")
        
        model.eval()
        all_preds = []
        all_labels = []

        logger.info("Starting evaluation of Vision Transformer model...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                logger.debug(f"Processed batch {batch_idx + 1}/{len(dataloader)}")

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Calculating evaluation metrics
        logger.info("Calculating evaluation metrics...")
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        report = classification_report(all_labels, all_preds)
        conf_matrix = confusion_matrix(all_labels, all_preds)

        # To log the evaluation results
        logger.info(f"Vision Transformer Model Accuracy: {accuracy}")
        logger.info(f"Vision Transformer Model F1 Score: {f1}")
        logger.info(f"Vision Transformer Model Precision: {precision}")
        logger.info(f"Vision Transformer Model Recall: {recall}")
        logger.info(f"Classification Report:\n{report}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")

        # Returning the evaluation metrics
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
    
    except Exception as e:
        logger.error(f"Error during Vision Transformer model evaluation: {e}", exc_info=True)
        raise
