import unittest
import torch
from torchsummary import summary
from src.models.cnn import CNNModel  # Replacing with actual model import
from src.models.transformer import TransformerModel  # Replacing with actual model import
from src.models.svm import SVMModel  # Replacing with actual model import
from src.models.bayesian import BayesianModel  # Replacing with actual model import
from src.models.vision_transformer import VisionTransformerModel  # Replacing with actual model import

class TestModel(unittest.TestCase):
    def setUp(self):
        """
        Setting up test variables and environment.
        """
        self.input_shape = (3, 224, 224)  # Example input shape for image data
        self.num_classes = 10  # Example number of classes

    def test_cnn_model(self):
        """
        Testing CNN model architecture.
        """
        model = CNNModel(num_classes=self.num_classes)
        model.eval()
        sample_input = torch.randn(1, *self.input_shape)
        output = model(sample_input)
        self.assertEqual(output.shape[1], self.num_classes)
        summary(model, self.input_shape)

    def test_transformer_model(self):
        """
        Testing Transformer model architecture.
        """
        model = TransformerModel(num_classes=self.num_classes)
        model.eval()
        sample_input = torch.randn(1, *self.input_shape)
        output = model(sample_input)
        self.assertEqual(output.shape[1], self.num_classes)
        summary(model, self.input_shape)

    def test_svm_model(self):
        """
        Testing SVM model architecture.
        """
        model = SVMModel(num_classes=self.num_classes)
        sample_input = torch.randn(1, *self.input_shape).numpy()
        output = model.predict(sample_input)
        self.assertEqual(len(output), 1)
        self.assertIn(output[0], range(self.num_classes))

    def test_bayesian_model(self):
        """
        Testing Bayesian model architecture.
        """
        model = BayesianModel(num_classes=self.num_classes)
        sample_input = torch.randn(1, *self.input_shape).numpy()
        output = model.predict(sample_input)
        self.assertEqual(len(output), 1)
        self.assertIn(output[0], range(self.num_classes))

    def test_vision_transformer_model(self):
        """
        Testing Vision Transformer model architecture.
        """
        model = VisionTransformerModel(num_classes=self.num_classes)
        model.eval()
        sample_input = torch.randn(1, *self.input_shape)
        output = model(sample_input)
        self.assertEqual(output.shape[1], self.num_classes)
        summary(model, self.input_shape)

if __name__ == "__main__":
    unittest.main()
