from OpenMpipi import *

a1lcd_WT_sequence = 'GSMASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQSSNFGPMKGGNFGGRSSGGSGGGGQYFAKPRNQGGYGGSSSSSSYGSGRRF'

a1lcd_chain = IDP("A1LCD_WT", a1lcd_WT_sequence)

chains = {
    a1lcd_chain:100
}

build_and_equilibrate_model(chains, equi_time=100*unit.nanosecond, pulling_time=20*unit.nanosecond,
                            long_side_scale_factor=1, 
                            target_density=0.1*unit.gram/unit.centimeter**3)
