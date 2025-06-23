"""
Core logging components for simulation tracking.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import uuid
from threading import RLock


@dataclass
class AgentAction:
    """Represents a single agent action."""
    timestamp: datetime
    agent_name: str
    action_type: str  # 'message', 'decision', 'strategy_update', etc.
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    utility_before: Optional[float] = None
    utility_after: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class UtilitySnapshot:
    """Represents utility values at a specific point in time."""
    timestamp: datetime
    round_number: int
    agent_name: str
    utility_value: float
    environment_state: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class AgentLogger:
    """Logger for individual agent actions and state changes."""
    
    def __init__(self, agent_name: str, simulation_id: str):
        self.agent_name = agent_name
        self.simulation_id = simulation_id
        self.actions: List[AgentAction] = []
        self.utility_history: List[UtilitySnapshot] = []
        self._lock = RLock()
        self.logger = logging.getLogger(f"agent.{agent_name}")
        
    def log_action(self, action_type: str, content: Any, 
                   metadata: Optional[Dict[str, Any]] = None,
                   utility_before: Optional[float] = None,
                   utility_after: Optional[float] = None) -> None:
        """Log an agent action."""
        with self._lock:
            action = AgentAction(
                timestamp=datetime.now(),
                agent_name=self.agent_name,
                action_type=action_type,
                content=content,
                metadata=metadata or {},
                utility_before=utility_before,
                utility_after=utility_after
            )
            self.actions.append(action)
            self.logger.info(f"Action logged: {action_type} - {content[:100] if isinstance(content, str) else content}")
    
    def log_utility(self, round_number: int, utility_value: float, 
                    environment_state: Optional[Dict[str, Any]] = None) -> None:
        """Log utility value at a specific round."""
        with self._lock:
            snapshot = UtilitySnapshot(
                timestamp=datetime.now(),
                round_number=round_number,
                agent_name=self.agent_name,
                utility_value=utility_value,
                environment_state=environment_state or {}
            )
            self.utility_history.append(snapshot)
            self.logger.info(f"Utility logged: Round {round_number} - Value {utility_value:.4f}")
    
    def get_utility_trend(self) -> List[Tuple[int, float]]:
        """Get utility values over rounds."""
        with self._lock:
            return [(s.round_number, s.utility_value) for s in self.utility_history]
    
    def get_actions_by_type(self, action_type: str) -> List[AgentAction]:
        """Get all actions of a specific type."""
        with self._lock:
            return [a for a in self.actions if a.action_type == action_type]


class MetricsCollector:
    """Collects and aggregates metrics across the simulation."""
    
    def __init__(self):
        self.metrics: Dict[str, List[Tuple[datetime, Any]]] = {}
        self._lock = RLock()
        
    def record(self, metric_name: str, value: Any) -> None:
        """Record a metric value."""
        with self._lock:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            self.metrics[metric_name].append((datetime.now(), value))
    
    def get_metric(self, metric_name: str) -> List[Tuple[datetime, Any]]:
        """Get all values for a specific metric."""
        with self._lock:
            return self.metrics.get(metric_name, [])
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all metrics."""
        summary = {}
        with self._lock:
            for metric_name, values in self.metrics.items():
                if values and all(isinstance(v[1], (int, float)) for v in values):
                    numeric_values = [v[1] for v in values]
                    summary[metric_name] = {
                        'count': len(numeric_values),
                        'mean': sum(numeric_values) / len(numeric_values),
                        'min': min(numeric_values),
                        'max': max(numeric_values),
                        'last': numeric_values[-1]
                    }
                else:
                    summary[metric_name] = {
                        'count': len(values),
                        'last': values[-1][1] if values else None
                    }
        return summary


class SimulationLogger:
    """Main logger for the entire simulation."""
    
    def __init__(self, simulation_id: Optional[str] = None, 
                 log_dir: Optional[Path] = None):
        self.simulation_id = simulation_id or str(uuid.uuid4())
        self.log_dir = log_dir or Path("logs") / self.simulation_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_time = datetime.now()
        self.agent_loggers: Dict[str, AgentLogger] = {}
        self.metrics = MetricsCollector()
        self.messages: List[Dict[str, Any]] = []
        self._round_number = 0
        self._lock = RLock()
        
        # Setup main logger
        self.logger = logging.getLogger(f"simulation.{self.simulation_id}")
        self._setup_file_logging()
        
    def _setup_file_logging(self):
        """Setup file logging handlers."""
        # Main simulation log
        main_handler = logging.FileHandler(self.log_dir / "simulation.log")
        main_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(main_handler)
        self.logger.setLevel(logging.INFO)
        
    def get_agent_logger(self, agent_name: str) -> AgentLogger:
        """Get or create an agent logger."""
        with self._lock:
            if agent_name not in self.agent_loggers:
                self.agent_loggers[agent_name] = AgentLogger(agent_name, self.simulation_id)
            return self.agent_loggers[agent_name]
    
    def log_message(self, agent_name: str, message: str, 
                    metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a conversation message."""
        with self._lock:
            msg_data = {
                'timestamp': datetime.now().isoformat(),
                'round': self._round_number,
                'agent': agent_name,
                'message': message,
                'metadata': metadata or {}
            }
            self.messages.append(msg_data)
            
            # Also log to agent logger
            agent_logger = self.get_agent_logger(agent_name)
            agent_logger.log_action('message', message, metadata)
    
    def increment_round(self) -> None:
        """Increment the round counter."""
        with self._lock:
            self._round_number += 1
            self.logger.info(f"Starting round {self._round_number}")
    
    def log_utility_update(self, agent_name: str, utility_value: float,
                          environment: Optional[Dict[str, Any]] = None) -> None:
        """Log utility update for an agent."""
        agent_logger = self.get_agent_logger(agent_name)
        agent_logger.log_utility(self._round_number, utility_value, environment)
        self.metrics.record(f"{agent_name}_utility", utility_value)
    
    def save_logs(self) -> None:
        """Save all logs to files."""
        # Save messages
        with open(self.log_dir / "messages.json", 'w') as f:
            json.dump(self.messages, f, indent=2)
        
        # Save agent actions
        for agent_name, agent_logger in self.agent_loggers.items():
            agent_data = {
                'agent_name': agent_name,
                'actions': [a.to_dict() for a in agent_logger.actions],
                'utility_history': [u.to_dict() for u in agent_logger.utility_history]
            }
            with open(self.log_dir / f"agent_{agent_name}.json", 'w') as f:
                json.dump(agent_data, f, indent=2)
        
        # Save metrics summary
        with open(self.log_dir / "metrics.json", 'w') as f:
            json.dump(self.metrics.get_summary(), f, indent=2)
        
        self.logger.info(f"Logs saved to {self.log_dir}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get simulation summary."""
        return {
            'simulation_id': self.simulation_id,
            'start_time': self.start_time.isoformat(),
            'duration': (datetime.now() - self.start_time).total_seconds(),
            'total_rounds': self._round_number,
            'total_messages': len(self.messages),
            'agents': list(self.agent_loggers.keys()),
            'metrics_summary': self.metrics.get_summary(),
            'log_directory': str(self.log_dir)
        }