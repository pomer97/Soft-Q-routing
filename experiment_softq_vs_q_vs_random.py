import os
from Utils import Statistics
from Utils.ml_flow_utils import preprocess_meta_data
from Environment import env
from Agents import Agent
from trainer import trainer, evaluator
import logging
import matplotlib
matplotlib.use('Agg')
import numpy as np

logger = logging.getLogger("logger")

envsLut = {
    'Q-Routing': env.dynetworkEnvQlearning,
    'Soft-Q-Routing': env.dynetworkEnvQlearning,  # Using same env as Q-Routing
    'Shortest-Path': env.dynetworkEnvShortestPath  # Changed from Random to Shortest-Path
}

agentsLut = {
    'Q-Routing': Agent.QAgent,
    'Soft-Q-Routing': Agent.SoftQAgent, 
    'Shortest-Path': Agent.QAgent  # Base agent class is fine since SP uses environment routing
}

trainerLut = {
    'Q-Routing': trainer.RLTabularTrainer,
    'Soft-Q-Routing': trainer.RLTabularTrainer,  # Using same trainer
    'Shortest-Path': None  # No training needed for shortest path
}

evaluatorLut = {
    'Q-Routing': evaluator.TabularRLRouterEvaluator,
    'Soft-Q-Routing': evaluator.TabularRLRouterEvaluator,
    'Shortest-Path': evaluator.RouterEvaluator  # Using basic evaluator for SP
}

class MultiAgentExperiment:
    def __init__(self, setting, experiment, agent_names):
        self.setting = setting
        self.experiment = experiment
        self.agent_names = agent_names
        self.results = {}
        self.agents = {}
        self.envs = {}
        self.trainers = {}
        self.evaluators = {}
        self.stats = {}
        self.paths = {}
        for name in agent_names:
            self.paths[name] = os.path.join(setting.result_dir, name)
            os.makedirs(self.paths[name] + '/train/', exist_ok=True)
            os.makedirs(self.paths[name] + '/test/', exist_ok=True)
            EnvClass = envsLut[name]
            AgentClass = agentsLut[name]
            TrainerClass = trainerLut[name]
            EvaluatorClass = evaluatorLut[name]
            env_instance = EnvClass(setting=setting, seed=setting["seed"], algorithm=name, rewardfun=setting["AGENT"]["rewardfunction"])
            state_space = env_instance.get_state_space_dim(setting)
            agent_instance = AgentClass(env_instance.dynetwork, setting, state_space, None)
            self.envs[name] = env_instance
            self.agents[name] = agent_instance
            self.stats[name] = {
                'train': Statistics.TrainQLStatisticsCollector(setting=setting, result_dir=self.paths[name] + '/train/', algorithms=[name]),
                'test': Statistics.TestStatisticsCollector(setting=setting, result_dir=self.paths[name] + '/test/', algorithms=[name])
            }
            if TrainerClass is not None:
                self.trainers[name] = TrainerClass(
                    time_steps=setting["Simulation"]["max_allowed_time_step_per_episode"],
                    TARGET_UPDATE=setting["Simulation"]["num_time_step_to_update_target_network"],
                    agent=agent_instance,
                    stat_collector=self.stats[name]["train"],
                    env=env_instance,
                    name=name,
                    writer=setting.train_writer,
                    experiment=experiment,
                    update_freq=setting["AGENT"]["learning_freq"]
                )
            else:
                self.trainers[name] = None
            self.evaluators[name] = EvaluatorClass(
                stat_collector=self.stats[name]["test"],
                env=env_instance,
                setting=setting,
                writer=setting.test_writer,
                update_freq=setting["AGENT"]["learning_freq"]
            )

    def run(self):
        for name in self.agent_names:
            logger.info(f"=== Training {name} ===")
            trainer = self.trainers[name]
            agent = self.agents[name]
            
            if trainer is not None and self.setting["AGENT"]["enable_train"] is True:
                for episode in range(self.setting["Simulation"]["training_episodes"]):
                    trainer.train(episode)
                    if episode % self.setting["AGENT"]["checkpoint_frequency"] == 0:
                        agent.save_agent(self.paths[name] + '/train/')
                        logger.info(f"Episode {episode}: {name} checkpoint saved")
                agent.save_agent(self.paths[name] + '/train/')
                
            logger.info(f"=== Evaluating {name} ===")
            self.evaluators[name].evaluate(agent)
            self.stats[name]['train'].plot_result(loud=False)
            self.stats[name]['train'].dump_statistics()
            self.stats[name]['test'].plot_result(loud=False)
            self.stats[name]['test'].dump_statistics()

if __name__ == "__main__":
    setting, args, temp_device, experiment = preprocess_meta_data()
    setting["capacity"] = setting["NETWORK"]["holding capacity"]
    setting["num_nodes"] = setting["NETWORK"]["number Basestation"] + setting["NETWORK"]["number user"]
    setting["num_bs"] = setting["NETWORK"]["number Basestation"]

    # Override settings for faster debugging - but keep reasonable values for packet flow
    setting["Simulation"]["training_episodes"] = 5  # Reduced from 100
    setting["Simulation"]["max_allowed_time_step_per_episode"] = 1000  # Reduced from 100000 but enough for packets
    setting["Simulation"]["test_allowed_time_step_per_episode"] = 500  # Reduced from 1000 but enough for testing
    setting["Simulation"]["num_time_step_to_update_target_network"] = 100  # Reduced from 1000
    setting["Simulation"]["test_trials_per_load"] = 1  # Reduced from 5
    setting["Simulation"]["test_network_load_min"] = 1  # Start at load 1 to ensure packets are generated
    setting["Simulation"]["test_network_load_max"] = 3  # Only test up to load 3

    # Compare Q-Routing against Soft-Q-Routing
    agent_names = ['Q-Routing', 'Soft-Q-Routing']
    exp = MultiAgentExperiment(setting=setting, experiment=experiment, agent_names=agent_names)
    exp.run()