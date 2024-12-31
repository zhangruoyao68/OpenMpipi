from OpenMpipi import *

RRR_seq = 10*'R'
RRR_chain = IDP('R6', RRR_seq)

RNA_seq = 10*'U'
RNA_chain = RNA('U8', RNA_seq)

chains = {
    RRR_chain:4000,
    RNA_chain:1000
}

build_and_equilibrate_model(chains, equi_time=500*unit.nanosecond, pulling_time=10*unit.nanosecond)
