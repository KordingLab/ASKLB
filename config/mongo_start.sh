#!/bin/bash

# command to start the mongo instance for asklb

sudo start-stop-daemon --start \
    --chuid mongodb:mongodb \
    --pidfile /var/run/mongodb_asklb.pid \
    --make-pidfile --exec /usr/bin/mongod \
    -- --config /etc/mongod_asklb.conf --fork