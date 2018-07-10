# Andromeda Reaper

The reaper is a very basic model for controlling mbots in the andromeda environment.
Essentially the bots move forward at a constant rate towards the first food that it has detected.
If there is poison detected closer than the food, the mbot will change it's heading.

## Use the reaper as a starting point

The reaper is designed to be the simplest possible model to help with building models for embod.ai
Feel free to fork the project and make modifications.

# Installing

## Requirements

You will need python 3.4 or above to use this library.

You can install requirements using
```
pip3 install -r requirments.txt
```

# Running

You will need create an agent and retrieve your api key before you can use this library.
You will need to create an embod.ai account to get these.

## Creating an agent and getting your API key

This is really easy...[**Click here to get started**](https://app.embod.ai/documentation/getting-started).

## Running

```python
python run.py -p [YOUR API KEY] -a [YOUR AGENT ID]
```

Once the agent is running you can view it's progress on the andromeda view page [here](https://app.embod.ai/andromeda/view)

You can also see the other agents that it is competing against here too!


## Support

Please contact support@embod.ai or join the [discord community](https://discordapp.com/invite/RPqqKr4)