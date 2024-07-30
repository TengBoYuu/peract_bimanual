# this script is for evaluating a given checkpoint.
#       bash scripts/eval.sh PERACT_BC  0 ${exp_name}


# some params specified by user
method=$1
# set the seed number
seed="0"
# set the gpu id for evaluation. we use one gpu for parallel evaluation.
eval_gpu=${2:-"0"}

test_demo_path="/mnt/disk_1/tengbo/bimanual_data/test"

addition_info="$(date +%Y%m%d)"
exp_name=${3:-"${method}_${addition_info}"}

starttime=`date +'%Y-%m-%d %H:%M:%S'`

camera=False
gripper_mode='BimanualDiscrete'
arm_action_mode='BimanualEndEffectorPoseViaPlanning'
action_mode='BimanualMoveArmThenGripper'

# tasks=[coordinated_push_box]
tasks=[bimanual_pick_laptop,bimanual_pick_plate,bimanual_straighten_rope,coordinated_lift_ball,coordinated_lift_tray,coordinated_push_box,coordinated_put_bottle_in_fridge,dual_push_buttons,handover_item,bimanual_sweep_to_dustpan,coordinated_take_tray_out_of_oven,handover_item_easy]
eval_type="last"

CUDA_VISIBLE_DEVICES=${eval_gpu} xvfb-run -a python eval.py method=$method \
    rlbench.task_name=${exp_name} \
    rlbench.demo_path=${test_demo_path} \
    framework.start_seed=${seed} \
    cinematic_recorder.enabled=${camera} \
    rlbench.gripper_mode=${gripper_mode} \
    rlbench.arm_action_mode=${arm_action_mode} \
    rlbench.action_mode=${action_mode} \
    rlbench.tasks=${tasks}  \
    framework.eval_type=${eval_type}


endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo "eclipsed time "$((end_seconds-start_seconds))"s"