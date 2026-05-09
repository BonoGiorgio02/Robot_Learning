import gymnasium as gym
import inspect
import pprint

env = gym.make("CartPole-v1")

print("ENV:")
print(env)

print("\nENV SPEC:")
print(env.spec)

print("\nSPEC DICT:")
pprint.pp(env.spec.__dict__)

print("\nWRAPPER CHAIN:")
current = env
level = 0

while True:
    print("\n" + "-" * 60)
    print("Level:", level)
    print("Object:", current)
    print("Type:", type(current))

    print("\nPublic attributes:")
    print([a for a in dir(current) if not a.startswith("_")])

    print("\nObject dict:")
    pprint.pp(current.__dict__)

    if not hasattr(current, "env"):
        break

    current = current.env
    level += 1

print("\nBASE ENV SOURCE FILE:")
print(inspect.getfile(type(env.unwrapped)))

print("\nBASE ENV INIT SOURCE:")
print(inspect.getsource(type(env.unwrapped).__init__))

# print(env)
# print(type(env))
# print(env.__dict__)

# base_env = env.unwrapped

# print("print env.spec to get registration metadata and also all its attributes with dir() and .__dict__:")
# print(type(env.spec))
# print(dir(env.spec))
# print(env.spec.__dict__)
# print(env.spec)
# print(env.spec.id)
# print(env.spec.max_episode_steps)
# print(env.spec.reward_threshold)

# print("")
# print("print dir(env) to get all available attributes and methods")
# print(dir(env.unwrapped))

# print("")
# print("print env.unwrapped.__dict__")
# print(env.unwrapped.__dict__)

# print("")
# print("Filtred attributes by dir(env)")
# attrs = [a for a in dir(env.unwrapped) if not a.startswith("_")]
# print(attrs)

# print("x_threshold:", base_env.x_threshold)
# print("theta_threshold_radians:", base_env.theta_threshold_radians)
# print("theta_threshold_degrees:", base_env.theta_threshold_radians * 180 / 3.141592653589793)

# print("max_episode_steps from spec:", env.spec.max_episode_steps)

# print(inspect.getsource(env.unwrapped.step))

env.close()