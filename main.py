import src.processing as process
from src.config import ModelConfig
from src.evaluation import evaluate_model
from src.training import NFLModel


if __name__ == "__main__":
    # Rebuild every feature stage with:
    # process.build_features()
    #
    # Download fresh play-by-play data and rebuild with:
    # process.build_features(download=True, seasons=range(2015, 2026))
    config = ModelConfig(
        use_market_history=True,
        stacking_strategy="kfold",
        market_history_features="all",
    )
    model = NFLModel(config)
    model.train()
    evaluate_model(model, "val")
    # model.get_feature_importances(stage=1)
    model.get_feature_importances(stage=2)
