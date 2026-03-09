Scenarios
=======================================

We collect two types of scenarios in this benchmark: **common scenarios** and **near-accident scenarios**. 
The former represent common behaviors of vehicles (e.g., lane change, turns, etc.) and serve as building blocks for more complicated scenarios. 
The latter reconstruct scenarios from DMV crash reports and serve as critical traffic situations.

Common Scenarios & Automated Scenario Generator
---------------------------------------------------
For common scenarios, we develop an automated Scenic program generator. By defining the map, the ego vehicle's behavior, and the attributes of surrounding agents (types, maneuvers, strategies, and spatial relations) in a JSON-like dictionary, the generator outputs a fully executable Scenic program. 

Here is an example scenario specification:

.. code-block:: json

    {
        "scenario": "test.scenic",
        "map": "../../maps/Town05.xodr",
        "ego": {
            "type": "CAR",
            "maneuver": "RIGHT_TURN"
        },
        "agents": {
            "agent1": {
                "type": "CAR",
                "maneuver": "LANE_FOLLOWING",
                "strategy": "conservative",
                "spatial_relation": "FASTER_LANE"
            },
            "agent2": {
                "type": "CAR",
                "maneuver": "LANE_FOLLOWING",
                "strategy": "aggressive",
                "spatial_relation": "AHEAD_OF"
            },
            "agent3": {
                "type": "PEDESTRIAN",
                "maneuver": "CROSS_STREET",
                "spatial_relation": "SIDEWALK"
            }
        }
    }

Generating Scenario Specs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To create a diverse set of scenarios, we provide a pipeline to automatically generate specifications using either a k-Center greedy algorithm or a random sampling approach. The pipeline outputs a ``.jsonl`` file containing the generated specs.

To execute the script, using the following command:

.. code-block:: shell

    python auto_scenario_generator.py generate_spec -f <FILE_NAME> [-v NUM_VEHICLES] [-p NUM_PED_AGENTS] [-n NUM_SCENARIOS] [-m {k_center,random}]

- -f: The output file name for the generated specifications (``*.jsonl``).
- -v: The number of surrounding vehicles to include.
- -p: The number of pedestrian agents to include.
- -n: The total number of scenario specifications to generate.
- -m: The generation method to use (``k_center`` or ``random``).

Generating Scenic Program from Specs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Once you have generated your scenario specifications, you can compile them into executable Scenic programs using the following command:

.. code-block:: shell

    python auto_scenario_generator.py generate_scenarios -f <FILE_NAME>

- -f: The input file name containing the scenario specifications (``*.jsonl``).

This will parse the specifications in your ``.jsonl`` file and output the corresponding ``.scenic`` program files.

For complete implementation details, please refer to ``scenarios/auto_scenario_generator.py``.

Common Scenarios
^^^^^^^^^^^^^^^^^^^^^
In the current version of the benchmark, we have generated 300 common scenarios under three different configurations using the k-Center greedy algorithm (100 for each configuration). The configurations are defined by the number of surrounding vehicles and pedestrian agents:

- **Configuration 1**: 2 surrounding vehicles and 0 pedestrians. The specs for this configuration are recorded in ``scenarios/common_specs/common_scenarios_20.jsonl``. The generated Scenic programs are stored as ``scenarios/common/common20_*.scenic``.
- **Configuration 2**: 2 surrounding vehicles and 1 pedestrians. The specs for this configuration are recorded in ``scenarios/common_specs/common_scenarios_21.jsonl``. The generated Scenic programs are stored as ``scenarios/common/common21_*.scenic``.
- **Configuration 3**: 3 surrounding vehicles and 0 pedestrians. The specs for this configuration are recorded in ``scenarios/common_specs/common_scenarios_30.jsonl``. The generated Scenic programs are stored as ``scenarios/common/common30_*.scenic``.

Users can also generate more scenarios by modifying the parameters in the spec generation command.

Near-Accident Scenarios & LLM-Assisted Scenic Code Generation
------------------------------------------------------------------
For near-accident scenarios, we provide an LLM-assisted pipeline to generate executable Scenic code directly from DMV crash reports written in natural language. Our prompting strategy guides the LLM by outlining the core structure of a Scenic program and providing several few-shot examples that map natural language crash reports to Scenic code. 

To execute the generation pipeline, follow these steps:

1. **Configure the API and Model**: Create a file named ``api.txt`` under ``scenarios/``. Insert your Google Gemini API key on the first line, and specify the target AI model (e.g., ``gemini-2.5-flash``) on the second line.
2. **Prepare the Input**: Create a text file containing the natural language description of your target scenario.
3. **Execute the Script**: Run the generation script using the following command:

.. code-block:: shell

    python scenicnl.py api.txt <input_file_path> <output_file_path>

where ``<input_file_path>`` is the path to the file containing your natural language description (e.g., ``scenario.txt``), and ``<output_file_path>`` is the desired output path for the generated Scenic program (e.g., ``scenario.scenic``).

.. note::
   The LLM-generated Scenic code may require manual review and refinement to ensure accuracy and completeness.

For complete implementation details, please refer to ``scenarios/scenicnl.py``.

Example Scenarios
^^^^^^^^^^^^^^^^^^^^^
We provide 27 example near-accident scenarios located in the ``scenarios/crash/`` directory. In each scenario (``crash_*.scenic``), the corresponding natural language description can be found in the header comments. 
The crash reports for these examples are curated from the "hard" tier of the `ScenicNL dataset <https://github.com/KE7/ScenarioNL-CA-AV-Crash>`_. Additional raw reports can be accessed through the `California DMV Autonomous Vehicle Collision Reports <https://www.dmv.ca.gov/portal/vehicle-industry-services/autonomous-vehicles/autonomous-vehicle-collision-reports/>`_ database.
