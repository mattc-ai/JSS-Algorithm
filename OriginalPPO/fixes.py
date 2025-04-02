# Save this as test_jssenv.py and run it with python test_jssenv.py

import os
import sys

# Add diagnostic information about Python and environment
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")

# Try to find a valid instance file
instance_paths = []

# Try to find JSSEnv installation
try:
    import JSSEnv
    print(f"JSSEnv found at: {JSSEnv.__file__}")
    
    # Try to get instances directory from JSSEnv
    if hasattr(JSSEnv, '__file__') and JSSEnv.__file__ is not None:
        instances_dir = os.path.join(os.path.dirname(JSSEnv.__file__), 'envs', 'instances')
        print(f"Looking for instances in: {instances_dir}")
        if os.path.exists(instances_dir):
            # List the first 5 instance files (if any)
            instance_files = os.listdir(instances_dir)[:5]
            print(f"Found {len(instance_files)} instance files. First few: {instance_files}")
            
            if instance_files:
                # Use the first instance file found
                instance_path = os.path.join(instances_dir, instance_files[0])
                print(f"Using instance file: {instance_path}")
            else:
                print("No instance files found in the expected directory.")
                instance_path = "NOT_FOUND"
        else:
            print(f"Instances directory not found at {instances_dir}")
            instance_path = "NOT_FOUND"
    else:
        print("JSSEnv.__file__ is None, can't determine installation path")
        instance_path = "NOT_FOUND"
        
except ImportError:
    print("JSSEnv package not found in Python path")
    instance_path = "NOT_FOUND"

# If we didn't find a valid instance, try some alternatives
if instance_path == "NOT_FOUND":
    # Try looking in common locations
    possible_locations = [
        "JSSEnv/JSSEnv/envs/instances",
        "JSSEnv/envs/instances",
        "../JSSEnv/JSSEnv/envs/instances",
        "../JSSEnv/envs/instances"
    ]
    
    for location in possible_locations:
        if os.path.exists(location):
            instance_files = os.listdir(location)
            if instance_files:
                instance_path = os.path.join(location, instance_files[0])
                print(f"Found instance at alternative location: {instance_path}")
                break
    
    if instance_path == "NOT_FOUND":
        print("Could not find any valid instance files. Using a dummy path for testing.")
        instance_path = "dummy_instance"

# Add this diagnostic code
try:
    print("\n--- Testing direct import of JssEnv ---")
    from JSSEnv.envs import JssEnv
    print("Successfully imported JssEnv class")
    
    print(f"Trying to create environment with instance_path: {instance_path}")
    env_test = JssEnv({'instance_path': instance_path})
    print("Successfully created environment instance directly")
    
    # Try to check the observation space
    print(f"Observation space: {env_test.observation_space}")
    print(f"Action space: {env_test.action_space}")
    
except Exception as e:
    print(f"Error importing environment directly: {e}")
    import traceback
    traceback.print_exc()

# Try using gym.make
try:
    print("\n--- Testing gym.make() ---")
    import gym
    
    # Ensure JSSEnv is imported before gym.make
    import JSSEnv
    env = gym.make('jss-v1', env_config={'instance_path': instance_path})
    print("Successfully created environment through gym.make()")
    
    # Try to reset the environment
    print("Resetting environment...")
    obs = env.reset()
    print(f"Reset successful. Observation keys: {obs.keys() if isinstance(obs, dict) else 'not a dict'}")
    
except Exception as e:
    print(f"Error creating environment through gym.make(): {e}")
    import traceback
    traceback.print_exc()
