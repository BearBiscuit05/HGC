source /root/xbinst_oem/F3_env_setup.sh xocl

workdir=$(cd $(dirname $0); pwd)

nohup make exe > log.txt 1>&2 &

XCL_EMULATION_MODE=sw_emu ./host ./xclbin/