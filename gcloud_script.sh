#!/bin/bash

INSTANCE_NAME=$(grep GCLOUD_INSTANCE_NAME .env.dev | cut -d '=' -f2)
ZONE_NAME=$(grep GCLOUD_INSTANCE_ZONE .env.dev | cut -d '=' -f2)


COMMAND="python3.6 $1 -s $2"

echo "START SCRIPT"

echo $INSTANCE_NAME
echo $ZONE_NAME
echo "Dataset: $1 and setting file: $2"

make clean
echo "LOCAL: folders cleaned"

gcloud compute instances start $INSTANCE_NAME --zone=$ZONE_NAME
returned=$(echo $?)

if [[ $returned == 0 ]]
then
    echo "VM: started"
    gcloud compute scp --recurse deepswarm/ $INSTANCE_NAME:~ --zone=$ZONE_NAME
    gcloud compute scp --recurse examples/ $INSTANCE_NAME:~ --zone=$ZONE_NAME
    gcloud compute scp --recurse settings/ $INSTANCE_NAME:~ --zone=$ZONE_NAME
    gcloud compute scp --recurse tests/ $INSTANCE_NAME:~ --zone=$ZONE_NAME
    gcloud compute scp requirements-dev.txt $INSTANCE_NAME:~ --zone=$ZONE_NAME
    gcloud compute scp Makefile $INSTANCE_NAME:~ --zone=$ZONE_NAME
    echo "VM: data transfered from local to instance"
    # gcloud compute ssh --zone=$ZONE_NAME $INSTANCE_NAME --command 'sudo apt install python3-pip'
    # gcloud compute ssh --zone=$ZONE_NAME $INSTANCE_NAME --command 'python3.6 -m pip install -r requirements-dev.txt'
    gcloud compute ssh --zone=$ZONE_NAME $INSTANCE_NAME --command "$COMMAND"
    echo "VM: commands executed"
    # gcloud compute scp -recurse $INSTANCE_NAME:saves/ gcloud_saves/
    # echo "VM: debug transfered from instance to local"
    gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE_NAME
    echo "VM: stopped"
    notify-send "DeepSwarm: gcloud VM run finished"
    echo "END SCRIPT"
else 
    echo "VM: NOT started"
    notify-send "Gcloud VM not started"
fi

