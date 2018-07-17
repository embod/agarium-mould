import argparse
from evolve import EvolvingController

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Mbot control using Simple heuristics.')
    parser.add_argument('-p', required=True, dest='api_key', help='Your embod.ai API key')
    parser.add_argument('-a', required=True, dest='agent_ids', nargs='+', help='The id of the agent you want to control')
    parser.add_argument('-H', default="wss://api.embod.ai/environment/6dde25ed-d76c-4456-9975-35205518c6e9", dest='host', help="The websocket host for the environment")

    args = parser.parse_args()

    controller = EvolvingController(args.api_key, args.agent_ids, args.host, 1000)

    controller.run()
