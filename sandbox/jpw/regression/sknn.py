from sknn.mlp import Regressor, Layer

def create_model():
    return Regressor(
        layers=[
        Layer("Rectifier", units=100),
        Layer("Linear")],
        learning_rate=0.02,
        n_iter=10
    )