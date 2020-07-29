#!/bin/bash

COMMAND="python3.6 $1 -s $2"

NB_INSTANCES=${3:-1}
echo $NB_INSTANCES
echo "START SCRIPT"

echo "Dataset: $1 and setting file: $2"

make clean
echo "LOCAL: folders cleaned"

iter=$NB_INSTANCES
for v in $(seq 1 $NB_INSTANCES)
do
    INSTANCE_NAME=$(grep GCLOUD_INSTANCE_NAME_$v .env | cut -d '=' -f2)
    ZONE_NAME=$(grep GCLOUD_INSTANCE_ZONE_$v .env | cut -d '=' -f2)
    echo $INSTANCE_NAME
    echo $ZONE_NAME

    gcloud compute instances start $INSTANCE_NAME --zone=$ZONE_NAME
    returned=$(echo $?)
    if [[ $returned == 0 ]]
    then
        echo "VM: $INSTANCE_NAME started"
        notify-send "DeepSwarm: $INSTANCE_NAME Nohup model launched"

        gcloud compute config-ssh
        sleep 120
        
        gcloud compute scp --recurse tests/ $INSTANCE_NAME:~ --zone=$ZONE_NAME
        gcloud compute scp --recurse examples/ $INSTANCE_NAME:~ --zone=$ZONE_NAME
        gcloud compute scp --recurse settings/ $INSTANCE_NAME:~ --zone=$ZONE_NAME
        gcloud compute scp --recurse deepswarm/ $INSTANCE_NAME:~ --zone=$ZONE_NAME

        gcloud compute scp gcloud_scripts/instance_run.sh $INSTANCE_NAME:~ --zone=$ZONE_NAME
        gcloud compute scp gcloud_scripts/init_instance.sh $INSTANCE_NAME:~ --zone=$ZONE_NAME

        gcloud compute scp gcloud_scripts/google_drive_sync.py $INSTANCE_NAME:~ --zone=$ZONE_NAME
        gcloud compute scp client_secrets.json $INSTANCE_NAME:~ --zone=$ZONE_NAME
        gcloud compute scp credentials.txt $INSTANCE_NAME:~ --zone=$ZONE_NAME

        gcloud compute scp requirements-dev-gpu.txt $INSTANCE_NAME:~ --zone=$ZONE_NAME
        gcloud compute scp Makefile $INSTANCE_NAME:~ --zone=$ZONE_NAME
        gcloud compute scp .env $INSTANCE_NAME:~ --zone=$ZONE_NAME
        echo "VM: data transfered from local to instance"

        nohup gcloud compute ssh --zone=$ZONE_NAME $INSTANCE_NAME --command "chmod +x instance_run.sh && ./instance_run.sh $1 $2 $4" &
        echo "VM: commands executed"
        
        notify-send "DeepSwarm: Nohup model launched"
        echo "END SCRIPT"
        exit
    fi
done 

echo "VM: NOT started"
notify-send "Gcloud VM not started"

