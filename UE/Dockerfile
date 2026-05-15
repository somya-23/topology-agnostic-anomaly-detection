# Base image
FROM srsran:split_8_zmq

RUN apt-get update && apt-get install -y libdwarf-dev iputils-ping iperf3 iperf

COPY ./libboost_program_options.so.1.74.0 /usr/lib/x86_64-linux-gnu/libboost_program_options.so.1.74.0
COPY ./srsRAN_4G/build /srsue_4G
COPY ./ping.sh ./ping.sh
RUN chmod +x ping.sh
