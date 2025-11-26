import numpy as np
from sklearn.ensemble import IsolationForest
from collections import deque

class AnomalyDetector:
    def __init__(self, history_size=100):
        self.history_size = history_size
        self.metrics_history = []
        self.model = IsolationForest(contamination=0.05, random_state=42)
        self.is_fitted = False
        self.anomalies = []
        
        # Thresholds for rule-based detection
        self.thresholds = {
            'density_change': 0.2,  # 20% sudden change
            'entropy_low': 1.0,     # Low entropy = high alignment
            'speed_high': 10.0      # High speed
        }

    def update(self, metrics):
        """
        Update with new metrics and return anomaly score/flag.
        """
        self.metrics_history.append(metrics)
        
        # Rule-based check
        is_rule_anomaly = self._check_rules(metrics)
        
        # ML-based check (needs history)
        is_ml_anomaly = False
        score = 0.0
        
        if len(self.metrics_history) > 20:
            # Prepare data for ML
            data = self._prepare_data()
            
            # Fit/Predict
            # In a real online scenario, we might want to retrain periodically or use a streaming algorithm.
            # Here we'll retrain every N frames or just predict if fitted.
            # For simplicity in this demo, we fit on available history every time (inefficient but works for small demo)
            try:
                self.model.fit(data)
                pred = self.model.predict([data[-1]])
                score = self.model.decision_function([data[-1]])[0]
                is_ml_anomaly = pred[0] == -1
            except:
                pass
            
        is_anomaly = is_rule_anomaly or is_ml_anomaly
        
        if is_anomaly:
            self.anomalies.append({
                'start_frame': metrics['frame'],
                'end_frame': metrics['frame'], # To be updated if continuous
                'score': float(score),
                'reason': 'Rule' if is_rule_anomaly else 'ML'
            })
            
        return is_anomaly, score

    def _check_rules(self, metrics):
        # Check 1: Sudden density change
        if len(self.metrics_history) > 1:
            prev = self.metrics_history[-2]
            if prev['density'] > 0:
                change = abs(metrics['density'] - prev['density']) / prev['density']
                if change > self.thresholds['density_change']:
                    return True
                    
        # Check 2: Low entropy (alignment)
        if metrics['directional_entropy'] < self.thresholds['entropy_low'] and metrics['directional_entropy'] > 0:
            return True
            
        return False

    def _prepare_data(self):
        # Extract features for ML
        features = []
        for m in self.metrics_history:
            features.append([
                m['density'],
                m['clustering'],
                m['avg_speed'],
                m['directional_entropy']
            ])
        return np.array(features)
