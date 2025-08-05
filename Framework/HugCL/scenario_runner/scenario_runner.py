from __future__ import print_function

import glob
import traceback
from datetime import datetime
from distutils.version import LooseVersion
import importlib
import inspect
import os
import signal
import sys
import time
import json
import pkg_resources
import pickle

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenario_manager import ScenarioManager
from srunner.tools.scenario_parser import ScenarioConfigurationParser

# Version of scenario_runner
VERSION = '0.9.15'

class ScenarioRunner(object):

    """
    This is the core scenario runner module. It is responsible for
    running (and repeating) a single scenario or a list of scenarios.
    """

    ego_vehicles = []
    idx = 0   # scenario index
    
    # Tunable parameters
    client_timeout = 10.0  # in seconds
    wait_for_world = 30.0  # in seconds
    frame_rate = 60.0      # in Hz

    # CARLA world and scenario handlers
    world = None
    manager = None
    finished = False
    agent_instance = None
    module_agent = None

    def __init__(self, port=2000, timeout=10.0):
        """
        Setup CARLA client and world
        Setup ScenarioManager
        """
        self.host = 'localhost'
        self.port = port
        self.timeout = timeout
        self.trafficManagerPort = 8000
        self.trafficManagerSeed = 0
        self.sync = True
        self.list = False
        self.agent = None
        self.agentConfig = ''
        self.output = False
        self.file = False
        self.junit = False
        self.json = False
        self.outputDir = ''
        self.configFile = ''
        self.addtionalScenario = ''
        self.debug = False
        self.reloadWorld = False
        self.record = False
        self.randomize = True
        self.repetitions = 1
        self.waitForEgo = True
        self.testing_path = 'scenario.pkl'
            
        with open(self.testing_path, 'rb') as file:
            self.scenario_list = pickle.load(file)
        
        if self.timeout:
            self.client_timeout = float(self.timeout)

        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        self.client = carla.Client(self.host, int(self.port))
        self.client.set_timeout(self.client_timeout)
        dist = pkg_resources.get_distribution("carla")
        if LooseVersion(dist.version) < LooseVersion('0.9.15'):
            raise ImportError("CARLA version 0.9.15 or newer required. CARLA version found: {}".format(dist))

        # Load agent if requested via command line args
        # If something goes wrong an exception will be thrown by importlib (ok here)
        if self.agent is not None:
            module_name = os.path.basename(self.agent).split('.')[0]
            sys.path.insert(0, os.path.dirname(self.agent))
            self.module_agent = importlib.import_module(module_name)

        # Create the ScenarioManager
        self.manager = ScenarioManager(self.debug, self.sync, self.timeout)
        
        # Create signal handler for SIGINT
        self._shutdown_requested = False
        if sys.platform != 'win32':
            signal.signal(signal.SIGHUP, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self._start_wall_time = datetime.now()

    def destroy(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """
        self._cleanup()
        if self.manager is not None:
            del self.manager
        if self.world is not None:
            del self.world
        if self.client is not None:
            del self.client

    def _signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        self._shutdown_requested = True
        if self.manager:
            self.manager.stop_scenario()
            self._cleanup()
            if not self.manager.get_running_status():
                raise RuntimeError("Timeout occurred during scenario execution")

    def _get_scenario_class_or_fail(self, scenario):
        """
        Get scenario class by scenario name
        If scenario is not supported or not found, exit script
        """
        # Path of all scenario at "srunner/scenarios" folder + the path of the additional scenario argument
        path = "./srunner/scenarios/*.py"
        scenarios_list = glob.glob(path.format(os.getenv('SCENARIO_RUNNER_ROOT', "./")))
        for scenario_file in scenarios_list:
            # Get their module
            module_name = os.path.basename(scenario_file).split('.')[0]
            sys.path.insert(0, os.path.dirname(scenario_file))
            scenario_module = importlib.import_module(module_name)

            # And their members of type class
            for member in inspect.getmembers(scenario_module, inspect.isclass):
                if scenario in member:
                    return member[1]

            # Remove unused Python paths
            sys.path.pop(0)

        print("Scenario '{}' not supported ... Exiting".format(scenario))
        sys.exit(-1)

    def _cleanup(self):
        """
        Remove and destroy all actors
        """
        if self.finished:
            return

        self.finished = True

        # Simulation still running and in synchronous mode?
        if self.world is not None and self.sync:
            try:
                # Reset to asynchronous mode
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
                self.client.get_trafficmanager(int(self.trafficManagerPort)).set_synchronous_mode(False)
            except RuntimeError:
                sys.exit(-1)

        self.manager.cleanup()

        CarlaDataProvider.cleanup()
        
        for i, _ in enumerate(self.ego_vehicles):
            if self.ego_vehicles[i]:
                if not self.waitForEgo and self.ego_vehicles[i] is not None and self.ego_vehicles[i].is_alive:
                    print("Destroying ego vehicle {}".format(self.ego_vehicles[i].id))
                    self.ego_vehicles[i].destroy()
                self.ego_vehicles[i] = None
        self.ego_vehicles = []

        if self.agent_instance:
            self.agent_instance.destroy()
            self.agent_instance = None

    def _prepare_ego_vehicles(self, ego_vehicles):
        """
        Spawn or update the ego vehicles
        """
        if not self.waitForEgo:
            for vehicle in ego_vehicles:
                self.ego_vehicles.append(CarlaDataProvider.request_new_actor(vehicle.model,
                                                                             vehicle.transform,
                                                                             vehicle.rolename,
                                                                             random_location=vehicle.random_location,
                                                                             color=vehicle.color,
                                                                             actor_category=vehicle.category))
                
        else:
            ego_vehicle_missing = True
            while ego_vehicle_missing:
                print("Waiting ego vehicle......")
                self.ego_vehicles = []
                ego_vehicle_missing = False
                for ego_vehicle in ego_vehicles:
                    ego_vehicle_found = False
                    carla_vehicles = CarlaDataProvider.get_world().get_actors().filter('vehicle.*')
                    for carla_vehicle in carla_vehicles:
                        if carla_vehicle.attributes['role_name'] == ego_vehicle.rolename:
                            ego_vehicle_found = True
                            self.ego_vehicles.append(carla_vehicle)
                            break
                    if not ego_vehicle_found:
                        ego_vehicle_missing = True
                        break
                    
            for i, _ in enumerate(self.ego_vehicles):
                self.ego_vehicles[i].set_transform(ego_vehicles[i].transform)
                self.ego_vehicles[i].set_target_velocity(carla.Vector3D())
                self.ego_vehicles[i].set_target_angular_velocity(carla.Vector3D())
                self.ego_vehicles[i].apply_control(carla.VehicleControl())
                CarlaDataProvider.register_actor(self.ego_vehicles[i], ego_vehicles[i].transform)

        # sync state
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def _analyze_scenario(self, config):
        """
        Provide feedback about success/failure of a scenario
        """

        # Create the filename
        current_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        junit_filename = None
        json_filename = None
        config_name = config.name
        if self.outputDir != '':
            config_name = os.path.join(self.outputDir, config_name)

        if self.junit:
            junit_filename = config_name + current_time + ".xml"
        if self.json:
            json_filename = config_name + current_time + ".json"
        filename = None
        if self.file:
            filename = config_name + current_time + ".txt"

        if not self.manager.analyze_scenario(self.output, filename, junit_filename, json_filename):
            print("All scenario tests were passed successfully!")
        else:
            print("Not all scenario tests were successful")
            if not (self.output or filename or junit_filename):
                print("Please run with --output for further information")

    def _record_criteria(self, criteria, name):
        """
        Filter the JSON serializable attributes of the criterias and
        dumps them into a file. This will be used by the metrics manager,
        in case the user wants specific information about the criterias.
        """
        file_name = name[:-4] + ".json"

        # Filter the attributes that aren't JSON serializable
        with open('temp.json', 'w', encoding='utf-8') as fp:

            criteria_dict = {}
            for criterion in criteria:

                criterion_dict = criterion.__dict__
                criteria_dict[criterion.name] = {}

                for key in criterion_dict:
                    if key != "name":
                        try:
                            key_dict = {key: criterion_dict[key]}
                            json.dump(key_dict, fp, sort_keys=False, indent=4)
                            criteria_dict[criterion.name].update(key_dict)
                        except TypeError:
                            pass

        os.remove('temp.json')

        # Save the criteria dictionary into a .json file
        with open(file_name, 'w', encoding='utf-8') as fp:
            json.dump(criteria_dict, fp, sort_keys=False, indent=4)

    def _load_and_wait_for_world(self, town, ego_vehicles=None):
        """
        Load a new CARLA world and provide data to CarlaDataProvider
        """

        if self.reloadWorld:
            self.world = self.client.load_world(town)
        else:
            # if the world should not be reloaded, wait at least until all ego vehicles are ready
            ego_vehicle_found = False
            if self.waitForEgo:
                print("Not all ego vehicles ready. Waiting ... ")
                while not ego_vehicle_found and not self._shutdown_requested:
                    vehicles = self.client.get_world().get_actors().filter('vehicle.*')
                    for ego_vehicle in ego_vehicles:
                        ego_vehicle_found = False
                        for vehicle in vehicles:
                            if vehicle.attributes['role_name'] == ego_vehicle.rolename:
                                ego_vehicle_found = True
                                break
                        if not ego_vehicle_found:
                            print("Not all ego vehicles ready. Waiting ... ")
                            time.sleep(1)
                            break

        self.world = self.client.get_world()
        
        actors = self.world.get_actors()
        light_actor_list = actors.filter('*traffic_light*')
        # close traffic light
        for light_actor in light_actor_list:
            light_actor.set_state(carla.TrafficLightState.Green)
            light_actor.freeze(True)

        if self.sync:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / self.frame_rate
            settings.no_rendering_mode = True
            self.world.apply_settings(settings)

        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)

        # Wait for the world to be ready
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()

        map_name = CarlaDataProvider.get_map().name.split('/')[-1]
        if map_name not in (town, "OpenDriveMap"):
            print("The CARLA server uses the wrong map: {}".format(map_name))
            print("This scenario requires to use map: {}".format(town))
            return False

        return True

    def _load_and_run_scenario(self, config):
        """
        Load and run the scenario given by config
        """
        result = False
        if not self._load_and_wait_for_world(config.town, config.ego_vehicles):
            self._cleanup()
            return False

        if self.agent:
            agent_class_name = self.module_agent.__name__.title().replace('_', '')
            try:
                self.agent_instance = getattr(self.module_agent, agent_class_name)(self.agentConfig)
                config.agent = self.agent_instance
            except Exception as e:          # pylint: disable=broad-except
                traceback.print_exc()
                print("Could not setup required agent due to {}".format(e))
                self._cleanup()
                return False

        CarlaDataProvider.set_traffic_manager_port(int(self.trafficManagerPort))
        tm = self.client.get_trafficmanager(int(self.trafficManagerPort))
        tm.set_random_device_seed(int(self.trafficManagerSeed))
        if self.sync:
            tm.set_synchronous_mode(True)

        # Prepare scenario
        try:
            self._prepare_ego_vehicles(config.ego_vehicles)
            scenario_class = self._get_scenario_class_or_fail(config.type)
            scenario = scenario_class(world=self.world,
                                      ego_vehicles=self.ego_vehicles,
                                      config=config,
                                      randomize=self.randomize,
                                      debug_mode=self.debug)
        except Exception as exception:                  # pylint: disable=broad-except
            print("The scenario cannot be loaded")
            traceback.print_exc()
            print(exception)
            self._cleanup()
            return False
        
        try:
            # Load scenario and run it
            self.manager.load_scenario(scenario, self.agent_instance)
            self.manager.run_scenario()

            # Remove all actors, stop the recorder and save all criterias (if needed)
            scenario.remove_all_actors()
            result = True
            
        except Exception as e:              # pylint: disable=broad-except
            traceback.print_exc()
            print(e)
            result = False
        self._cleanup()
        return result

    def _run_scenarios(self, scenario_name):
        """
        Run conventional scenarios (e.g. implemented using the Python API of ScenarioRunner)
        """
        result = False
        # Load the scenario configurations provided in the config file
        self.scenario = scenario_name
        scenario_configurations = ScenarioConfigurationParser.parse_scenario_configuration(
            self.scenario,
            self.configFile)

        if not scenario_configurations:
            print("Configuration for scenario {} cannot be found!".format(self.scenario))
            return result
        
        # Execute each configuration
        for config in scenario_configurations:
            self.finished = False
            result = self._load_and_run_scenario(config)
            self._cleanup()
        return result
    
    def run(self):
        """
        Run all scenarios according to provided commandline args
        """
        result = True
        start_epoch = 0
        for idx in range(start_epoch, self.repetitions):
            scenario_name = self.scenario_selection(idx)
            print("\nPreparing scenario: {} ({})".format(scenario_name, idx))
            result = self._run_scenarios(scenario_name)
        print("No more scenarios .... Exiting")
        return result

    def scenario_selection(self, idx):
        scenario_name = self.scenario_list[idx]['scenario']
        return scenario_name


def main():
    scenario_runner = None
    result = True
    
    try:
        scenario_runner = ScenarioRunner()
        result = scenario_runner.run()
    except Exception:   # pylint: disable=broad-except
        traceback.print_exc()
        
    finally:
        if scenario_runner is not None:
            scenario_runner.destroy()
            del scenario_runner
    return not result
    

if __name__ == "__main__":
    main()
