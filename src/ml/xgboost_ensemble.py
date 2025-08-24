#!/usr/bin/env python3
"""
XGBoost Ensemble Layer for Kimchi Premium Prediction
Task #16: XGBoost Ensemble with LSTM outputs
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score

from src.utils.logger import logger


class XGBoostEnsemble:
    """
    XGBoost ensemble model for kimchi premium prediction
    Can work standalone or ensemble with LSTM outputs
    """
    
    def __init__(
        self,
        model_type: str = "regression",  # "regression" or "classification"
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        use_gpu: bool = False,
        **kwargs
    ):
        """
        Initialize XGBoost ensemble
        
        Args:
            model_type: Type of model ("regression" or "classification")
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            subsample: Subsample ratio
            colsample_bytree: Column subsample ratio
            random_state: Random seed
            use_gpu: Whether to use GPU
            **kwargs: Additional XGBoost parameters
        """
        self.model_type = model_type
        self.random_state = random_state
        
        # XGBoost parameters
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state,
            'n_jobs': -1,
            'verbosity': 0,
        }
        
        # GPU support
        if use_gpu:
            self.params['tree_method'] = 'gpu_hist'
            self.params['predictor'] = 'gpu_predictor'
        
        # Update with additional parameters
        self.params.update(kwargs)
        
        # Create model
        if model_type == "regression":
            self.params['objective'] = 'reg:squarederror'
            self.params['eval_metric'] = 'rmse'
            self.model = xgb.XGBRegressor(**self.params)
        else:
            self.params['objective'] = 'binary:logistic'
            self.params['eval_metric'] = 'auc'
            self.model = xgb.XGBClassifier(**self.params)
        
        # Feature importance
        self.feature_importance_ = None
        self.feature_names_ = None
        
        # SHAP explainer
        self.explainer = None
        
        logger.info(f"XGBoost {model_type} model initialized with {n_estimators} trees")
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        eval_set: Optional[List[Tuple]] = None,
        early_stopping_rounds: int = 10,
        verbose: bool = False
    ):
        """
        Train XGBoost model
        
        Args:
            X: Training features
            y: Training targets
            eval_set: Validation set for early stopping
            early_stopping_rounds: Early stopping patience
            verbose: Whether to print training progress
        """
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Train model
        if eval_set is not None:
            # Set early stopping in model params
            self.model.set_params(
                early_stopping_rounds=early_stopping_rounds,
                eval_metric='rmse' if self.model_type == 'regression' else 'logloss'
            )
            self.model.fit(
                X, y,
                eval_set=eval_set,
                verbose=verbose
            )
            logger.info(f"Model trained with best iteration: {self.model.best_iteration}")
        else:
            self.model.fit(X, y, verbose=verbose)
            logger.info(f"Model trained with {self.model.n_estimators} trees")
        
        # Get feature importance
        self.feature_importance_ = self.model.feature_importances_
        
        # Create SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
    
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        return_confidence: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions
        
        Args:
            X: Features to predict
            return_confidence: Whether to return confidence intervals
            
        Returns:
            predictions: Model predictions
            confidence: Confidence intervals (if requested)
        """
        predictions = self.model.predict(X)
        
        if return_confidence:
            # Get prediction intervals using quantile regression
            confidence = self._get_confidence_intervals(X)
            return predictions, confidence
        
        return predictions
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict probabilities (for classification)
        
        Args:
            X: Features to predict
            
        Returns:
            Prediction probabilities
        """
        if self.model_type != "classification":
            raise ValueError("predict_proba only available for classification models")
        
        return self.model.predict_proba(X)
    
    def ensemble_predict(
        self,
        X_features: Union[np.ndarray, pd.DataFrame],
        lstm_outputs: np.ndarray,
        weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Ensemble prediction combining XGBoost and LSTM
        
        Args:
            X_features: Original features for XGBoost
            lstm_outputs: LSTM model predictions
            weights: Ensemble weights {"xgboost": 0.5, "lstm": 0.5}
            
        Returns:
            Ensemble predictions
        """
        if weights is None:
            weights = {"xgboost": 0.5, "lstm": 0.5}
        
        # Normalize weights
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        # Get XGBoost predictions
        xgb_preds = self.predict(X_features)
        
        # Reshape LSTM outputs if needed
        if lstm_outputs.ndim > 1:
            lstm_preds = lstm_outputs.mean(axis=1)  # Average over time steps
        else:
            lstm_preds = lstm_outputs
        
        # Weighted ensemble
        ensemble_preds = (
            weights["xgboost"] * xgb_preds +
            weights["lstm"] * lstm_preds
        )
        
        logger.info(f"Ensemble prediction with weights: {weights}")
        
        return ensemble_preds
    
    def get_feature_importance(
        self,
        importance_type: str = "gain",
        top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get feature importance
        
        Args:
            importance_type: Type of importance ("gain", "weight", "cover")
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance_ is None:
            raise ValueError("Model must be trained first")
        
        # Get importance by type
        if importance_type == "gain":
            importance = self.model.get_booster().get_score(importance_type='gain')
        elif importance_type == "weight":
            importance = self.model.get_booster().get_score(importance_type='weight')
        elif importance_type == "cover":
            importance = self.model.get_booster().get_score(importance_type='cover')
        else:
            importance = dict(zip(self.feature_names_, self.feature_importance_))
        
        # Convert to DataFrame
        importance_df = pd.DataFrame(
            list(importance.items()),
            columns=['feature', 'importance']
        ).sort_values('importance', ascending=False)
        
        if top_n is not None:
            importance_df = importance_df.head(top_n)
        
        return importance_df
    
    def get_shap_values(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        check_additivity: bool = False
    ) -> np.ndarray:
        """
        Get SHAP values for interpretability
        
        Args:
            X: Features to explain
            check_additivity: Whether to check SHAP additivity
            
        Returns:
            SHAP values
        """
        if self.explainer is None:
            raise ValueError("Model must be trained first")
        
        shap_values = self.explainer.shap_values(X, check_additivity=check_additivity)
        
        return shap_values
    
    def explain_prediction(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        index: int = 0
    ) -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP
        
        Args:
            X: Features
            index: Index of sample to explain
            
        Returns:
            Dictionary with explanation details
        """
        # Get prediction
        if isinstance(X, pd.DataFrame):
            sample = X.iloc[index:index+1]
        else:
            sample = X[index:index+1]
        
        prediction = self.predict(sample)[0]
        
        # Get SHAP values
        shap_values = self.get_shap_values(sample)[0]
        
        # Get base value
        base_value = self.explainer.expected_value
        
        # Create explanation
        explanation = {
            'prediction': float(prediction),
            'base_value': float(base_value),
            'shap_values': shap_values.tolist(),
            'feature_names': self.feature_names_,
            'feature_values': sample.values[0].tolist() if isinstance(sample, pd.DataFrame) else sample[0].tolist()
        }
        
        # Add feature contributions
        contributions = []
        for i, (name, shap_val) in enumerate(zip(self.feature_names_, shap_values)):
            contributions.append({
                'feature': name,
                'value': explanation['feature_values'][i],
                'shap': float(shap_val),
                'contribution': float(shap_val) * 100 / abs(prediction - base_value) if prediction != base_value else 0
            })
        
        # Sort by absolute SHAP value
        contributions.sort(key=lambda x: abs(x['shap']), reverse=True)
        explanation['top_contributors'] = contributions[:10]
        
        return explanation
    
    def _get_confidence_intervals(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        alpha: float = 0.05
    ) -> np.ndarray:
        """
        Get prediction confidence intervals using quantile regression
        
        Args:
            X: Features
            alpha: Significance level (0.05 for 95% CI)
            
        Returns:
            Confidence intervals [lower, upper]
        """
        # Train quantile regressors
        lower_model = xgb.XGBRegressor(
            **{**self.params, 'objective': 'reg:quantileerror', 'quantile_alpha': alpha/2}
        )
        upper_model = xgb.XGBRegressor(
            **{**self.params, 'objective': 'reg:quantileerror', 'quantile_alpha': 1-alpha/2}
        )
        
        # Use the same training data (stored during fit)
        # This is a simplified version - in production, store training data
        predictions = self.predict(X)
        
        # Estimate intervals based on prediction variance
        # This is a heuristic - better to use proper quantile regression
        std_estimate = np.std(predictions) * 0.1  # Rough estimate
        z_score = stats.norm.ppf(1 - alpha/2)
        
        lower = predictions - z_score * std_estimate
        upper = predictions + z_score * std_estimate
        
        return np.column_stack([lower, upper])
    
    def cross_validate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        cv: int = 5,
        scoring: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Targets
            cv: Number of folds
            scoring: Scoring metric
            
        Returns:
            Cross-validation scores
        """
        if scoring is None:
            scoring = 'r2' if self.model_type == 'regression' else 'f1'
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        
        return {
            'scores': scores.tolist(),
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'scoring': scoring
        }
    
    def hyperparameter_search(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        param_grid: Optional[Dict] = None,
        cv: int = 3,
        scoring: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter search using GridSearchCV
        
        Args:
            X: Features
            y: Targets
            param_grid: Parameter grid for search
            cv: Number of CV folds
            scoring: Scoring metric
            
        Returns:
            Best parameters and scores
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.6, 0.8, 1.0]
            }
        
        if scoring is None:
            scoring = 'r2' if self.model_type == 'regression' else 'f1'
        
        # Create base model
        if self.model_type == "regression":
            base_model = xgb.XGBRegressor(random_state=self.random_state)
        else:
            base_model = xgb.XGBClassifier(random_state=self.random_state)
        
        # Grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.model.set_params(**grid_search.best_params_)
        
        logger.info(f"Best parameters found: {grid_search.best_params_}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': float(grid_search.best_score_),
            'cv_results': pd.DataFrame(grid_search.cv_results_).to_dict()
        }
    
    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X: Test features
            y: Test targets
            
        Returns:
            Evaluation metrics
        """
        predictions = self.predict(X)
        
        if self.model_type == "regression":
            metrics = {
                'mse': float(mean_squared_error(y, predictions)),
                'rmse': float(np.sqrt(mean_squared_error(y, predictions))),
                'mae': float(mean_absolute_error(y, predictions)),
                'r2': float(r2_score(y, predictions)),
                'mape': float(np.mean(np.abs((y - predictions) / y)) * 100) if not np.any(y == 0) else None
            }
        else:
            # For classification
            metrics = {
                'accuracy': float(accuracy_score(y, predictions)),
                'precision': float(precision_score(y, predictions, average='weighted')),
                'recall': float(recall_score(y, predictions, average='weighted')),
                'f1': float(f1_score(y, predictions, average='weighted'))
            }
        
        return metrics
    
    def save_model(self, filepath: str):
        """
        Save model to file
        
        Args:
            filepath: Path to save model
        """
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'params': self.params,
            'feature_names': self.feature_names_,
            'feature_importance': self.feature_importance_
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model from file
        
        Args:
            filepath: Path to load model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.params = model_data['params']
        self.feature_names_ = model_data['feature_names']
        self.feature_importance_ = model_data['feature_importance']
        
        # Recreate SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        
        logger.info(f"Model loaded from {filepath}")


class HybridEnsemble:
    """
    Hybrid ensemble combining LSTM and XGBoost for kimchi premium prediction
    """
    
    def __init__(
        self,
        lstm_model: Optional[Any] = None,
        xgboost_model: Optional[XGBoostEnsemble] = None,
        ensemble_method: str = "weighted",  # "weighted", "stacking", "voting"
        meta_learner: Optional[Any] = None
    ):
        """
        Initialize hybrid ensemble
        
        Args:
            lstm_model: Trained LSTM model
            xgboost_model: Trained XGBoost model
            ensemble_method: Method for combining predictions
            meta_learner: Meta-learner for stacking
        """
        self.lstm_model = lstm_model
        self.xgboost_model = xgboost_model
        self.ensemble_method = ensemble_method
        self.meta_learner = meta_learner
        
        # Ensemble weights (learned during training)
        self.weights = {"lstm": 0.5, "xgboost": 0.5}
        
        logger.info(f"Hybrid ensemble initialized with {ensemble_method} method")
    
    def fit_ensemble_weights(
        self,
        X_features: np.ndarray,
        X_sequences: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2
    ):
        """
        Learn optimal ensemble weights
        
        Args:
            X_features: Features for XGBoost
            X_sequences: Sequences for LSTM
            y: Target values
            validation_split: Validation split ratio
        """
        # Split data
        n_samples = len(y)
        n_train = int(n_samples * (1 - validation_split))
        
        X_features_train, X_features_val = X_features[:n_train], X_features[n_train:]
        X_sequences_train, X_sequences_val = X_sequences[:n_train], X_sequences[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        
        # Get predictions from both models
        lstm_preds = self.lstm_model.predict(X_sequences_val)
        if lstm_preds.ndim > 2:  # If shape is (batch, time, features)
            lstm_preds = lstm_preds.mean(axis=(1, 2))  # Average over time and features
        elif lstm_preds.ndim > 1:  # If shape is (batch, time) or (batch, features)
            lstm_preds = lstm_preds.mean(axis=1)  # Average over time or features
        
        # Ensure lstm_preds is 1D
        lstm_preds = lstm_preds.flatten()[:len(y_val)]
        
        xgb_preds = self.xgboost_model.predict(X_features_val)
        
        # Find optimal weights using grid search
        best_score = float('inf')
        best_weights = self.weights
        
        for lstm_weight in np.arange(0, 1.1, 0.1):
            xgb_weight = 1 - lstm_weight
            
            ensemble_preds = lstm_weight * lstm_preds + xgb_weight * xgb_preds
            score = mean_squared_error(y_val, ensemble_preds)
            
            if score < best_score:
                best_score = score
                best_weights = {"lstm": lstm_weight, "xgboost": xgb_weight}
        
        self.weights = best_weights
        logger.info(f"Optimal ensemble weights: {self.weights}")
    
    def predict(
        self,
        X_features: np.ndarray,
        X_sequences: np.ndarray,
        return_components: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Make ensemble predictions
        
        Args:
            X_features: Features for XGBoost
            X_sequences: Sequences for LSTM
            return_components: Whether to return individual model predictions
            
        Returns:
            Ensemble predictions or dictionary with all components
        """
        # Get LSTM predictions
        lstm_preds = self.lstm_model.predict(X_sequences)
        if lstm_preds.ndim > 2:  # If shape is (batch, time, features)
            lstm_preds = lstm_preds.mean(axis=(1, 2))  # Average over time and features
        elif lstm_preds.ndim > 1:  # If shape is (batch, time) or (batch, features)
            lstm_preds = lstm_preds.mean(axis=1)  # Average over time or features
        
        # Ensure lstm_preds is 1D
        lstm_preds = lstm_preds.flatten()
        
        # Get XGBoost predictions
        xgb_preds = self.xgboost_model.predict(X_features)
        
        # Combine predictions
        if self.ensemble_method == "weighted":
            ensemble_preds = (
                self.weights["lstm"] * lstm_preds +
                self.weights["xgboost"] * xgb_preds
            )
        elif self.ensemble_method == "voting":
            # Simple average for voting
            ensemble_preds = (lstm_preds + xgb_preds) / 2
        elif self.ensemble_method == "stacking":
            # Use meta-learner
            if self.meta_learner is None:
                raise ValueError("Meta-learner required for stacking")
            
            stack_features = np.column_stack([lstm_preds, xgb_preds])
            ensemble_preds = self.meta_learner.predict(stack_features)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        if return_components:
            return {
                'ensemble': ensemble_preds,
                'lstm': lstm_preds,
                'xgboost': xgb_preds,
                'weights': self.weights
            }
        
        return ensemble_preds
    
    def explain_ensemble(
        self,
        X_features: np.ndarray,
        X_sequences: np.ndarray,
        index: int = 0
    ) -> Dict[str, Any]:
        """
        Explain ensemble prediction
        
        Args:
            X_features: Features for XGBoost
            X_sequences: Sequences for LSTM
            index: Sample index to explain
            
        Returns:
            Comprehensive explanation
        """
        # Get all predictions
        predictions = self.predict(
            X_features[index:index+1],
            X_sequences[index:index+1],
            return_components=True
        )
        
        # Get XGBoost explanation
        xgb_explanation = self.xgboost_model.explain_prediction(X_features, index)
        
        # Create ensemble explanation
        explanation = {
            'ensemble_prediction': float(predictions['ensemble'][0]),
            'lstm_prediction': float(predictions['lstm'][0]),
            'xgboost_prediction': float(predictions['xgboost'][0]),
            'weights': self.weights,
            'xgboost_explanation': xgb_explanation,
            'contribution': {
                'lstm': float(self.weights['lstm'] * predictions['lstm'][0]),
                'xgboost': float(self.weights['xgboost'] * predictions['xgboost'][0])
            }
        }
        
        return explanation