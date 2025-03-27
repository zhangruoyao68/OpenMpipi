from OpenMpipi import *

a1lcd_WT_sequence = 'GSMASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQSSNFGPMKGGNFGGRSSGGSGGGGQYFAKPRNQGGYGGSSSSSSYGSGRRF'

a1lcd_chain = IDP("A1LCD_WT", a1lcd_WT_sequence)

chains = {
    a1lcd_chain:100
}

build_and_equilibrate_model(chains, equi_time=200*unit.nanosecond, pulling_time=20*unit.nanosecond)
