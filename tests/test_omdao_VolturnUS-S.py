import os
from pathlib import Path

import openmdao.api as om
import yaml

from raft.omdao_raft import RAFT_Group

'''
This example runs RAFT given the exact inputs and options that WEIS provides RAFT in 15_RAFT_Studies
To generate the inputs and options, set the DEBUG_OMDAO flag to true in omdao_raft.py
To debug a specific instance of the opendmdao RAFT run, set the options and input files below using absolute paths
'''

weis_options_file = 'weis_options.yaml'
weis_inputs_file = 'weis_inputs.yaml'

# -----------------------------------
# OMDAO
# -----------------------------------

def test_omdao_raft():
    this_dir = os.path.dirname(__file__)

    # Load options directly generated in WEIS
    with open(os.path.join(this_dir,'test_data',weis_options_file)) as file:
        opt = yaml.load(file, Loader=yaml.FullLoader)

    prob = om.Problem()
    prob.model = RAFT_Group(modeling_options=opt['modeling_options'],
                            analysis_options=opt['analysis_options'],
                            turbine_options=opt['turbine_options'],
                            mooring_options=opt['mooring_options'],
                            member_options=opt['member_options'])
    prob.setup()

    # -------------------------
    # inputs
    # -------------------------
    # Load options directly generated in WEIS
    with open(os.path.join(this_dir,'test_data',weis_inputs_file)) as file:
        inputs = yaml.load(file, Loader=yaml.FullLoader)

    for key, val in inputs.items():
        prob[key] = val


    prob.run_model()


def test_save_designs_handles_concurrent_directory_creation(tmp_path, monkeypatch):
    options_file = Path(__file__).with_name("test_data") / weis_options_file
    with options_file.open(encoding="utf-8") as file:
        options = yaml.safe_load(file)

    options["modeling_options"]["save_designs"] = True
    options["analysis_options"]["general"]["folder_output"] = str(tmp_path)
    problem = om.Problem(model=RAFT_Group(**options), reports=False)

    design_dir = tmp_path / "raft_designs"
    makedirs = os.makedirs

    def makedirs_with_peer(path, *args, **kwargs):
        if Path(path) == design_dir:
            # Simulate another rank creating the directory just before this one.
            design_dir.mkdir(exist_ok=True)
        return makedirs(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, "makedirs", makedirs_with_peer)
        problem.setup()

    assert design_dir.is_dir()


def test_omdao_downwind_raft():
    this_dir = os.path.dirname(__file__)

    # Load options directly generated in WEIS
    with open(os.path.join(this_dir,'test_data',weis_options_file)) as file:
        opt = yaml.load(file, Loader=yaml.FullLoader)

    prob = om.Problem()
    prob.model = RAFT_Group(modeling_options=opt['modeling_options'],
                            analysis_options=opt['analysis_options'],
                            turbine_options=opt['turbine_options'],
                            mooring_options=opt['mooring_options'],
                            member_options=opt['member_options'])
    prob.setup()

    # -------------------------
    # inputs
    # -------------------------
    # Load options directly generated in WEIS
    with open(os.path.join(this_dir,'test_data',weis_inputs_file)) as file:
        inputs = yaml.load(file, Loader=yaml.FullLoader)

    for key, val in inputs.items():
        prob[key] = val

    prob['rotor_orientation'] = 'downwind'  # Set the rotor orientation to downwind

    prob.run_model()

    # check if the model ran successfully in downwind configuration
    assert prob.model.raft._outputs['dw_check'] == 1

if __name__=="__main__":
    test_omdao_raft()
    test_omdao_downwind_raft()
