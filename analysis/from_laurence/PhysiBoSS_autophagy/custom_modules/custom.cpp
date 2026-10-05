/*
###############################################################################
# If you use PhysiCell in your project, please cite PhysiCell and the version #
# number, such as below:                                                      #
#                                                                             #
# We implemented and solved the model using PhysiCell (Version x.y.z) [1].    #
#                                                                             #
# [1] A Ghaffarizadeh, R Heiland, SH Friedman, SM Mumenthaler, and P Macklin, #
#     PhysiCell: an Open Source Physics-Based Cell Simulator for Multicellu-  #
#     lar Systems, PLoS Comput. Biol. 14(2): e1005991, 2018                   #
#     DOI: 10.1371/journal.pcbi.1005991                                       #
#                                                                             #
# See VERSION.txt or call get_PhysiCell_version() to get the current version  #
#     x.y.z. Call display_citations() to get detailed information on all cite-#
#     able software used in your PhysiCell application.                       #
#                                                                             #
# Because PhysiCell extensively uses BioFVM, we suggest you also cite BioFVM  #
#     as below:                                                               #
#                                                                             #
# We implemented and solved the model using PhysiCell (Version x.y.z) [1],    #
# with BioFVM [2] to solve the transport equations.                           #
#                                                                             #
# [1] A Ghaffarizadeh, R Heiland, SH Friedman, SM Mumenthaler, and P Macklin, #
#     PhysiCell: an Open Source Physics-Based Cell Simulator for Multicellu-  #
#     lar Systems, PLoS Comput. Biol. 14(2): e1005991, 2018                   #
#     DOI: 10.1371/journal.pcbi.1005991                                       #
#                                                                             #
# [2] A Ghaffarizadeh, SH Friedman, and P Macklin, BioFVM: an efficient para- #
#     llelized diffusive transport solver for 3-D biological simulations,     #
#     Bioinformatics 32(8): 1256-8, 2016. DOI: 10.1093/bioinformatics/btv730  #
#                                                                             #
###############################################################################
#                                                                             #
# BSD 3-Clause License (see https://opensource.org/licenses/BSD-3-Clause)     #
#                                                                             #
# Copyright (c) 2015-2018, Paul Macklin and the PhysiCell Project             #
# All rights reserved.                                                        #
#                                                                             #
# Redistribution and use in source and binary forms, with or without          #
# modification, are permitted provided that the following conditions are met: #
#                                                                             #
# 1. Redistributions of source code must retain the above copyright notice,   #
# this list of conditions and the following disclaimer.                       #
#                                                                             #
# 2. Redistributions in binary form must reproduce the above copyright        #
# notice, this list of conditions and the following disclaimer in the         #
# documentation and/or other materials provided with the distribution.        #
#                                                                             #
# 3. Neither the name of the copyright holder nor the names of its            #
# contributors may be used to endorse or promote products derived from this   #
# software without specific prior written permission.                         #
#                                                                             #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" #
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE   #
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE  #
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE   #
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR         #
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF        #
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS    #
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN     #
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)     #
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE  #
# POSSIBILITY OF SUCH DAMAGE.                                                 #
#                                                                             #
###############################################################################
*/

#include "custom.h"
#include "../BioFVM/BioFVM.h"  
using namespace BioFVM;

// declare cell definitions here 

std::vector<bool> nodes;

void create_cell_types( void )
{
	// set the random seed 
	if (parameters.ints.find_index("random_seed") != -1)
	{
		SeedRandom(parameters.ints("random_seed"));
	}
	
	/* 
	   Put any modifications to default cell definition here if you 
	   want to have "inherited" by other cell types. 
	   
	   This is a good place to set default functions. 
	*/ 

	initialize_default_cell_definition(); 
	cell_defaults.phenotype.secretion.sync_to_microenvironment( &microenvironment ); 

	cell_defaults.functions.volume_update_function = standard_volume_update_function;
	cell_defaults.functions.update_velocity = NULL;
	cell_defaults.functions.update_phenotype = NULL; 
	cell_defaults.functions.update_migration_bias = NULL; 
	cell_defaults.functions.pre_update_intracellular = pre_update_intracellular; 
	cell_defaults.functions.post_update_intracellular = post_update_intracellular; 
	cell_defaults.functions.custom_cell_rule = NULL; 
	
	cell_defaults.functions.add_cell_basement_membrane_interactions = NULL; 
	cell_defaults.functions.calculate_distance_to_membrane = NULL; 
	
	cell_defaults.custom_data.add_variable(parameters.strings("node_to_visualize"), "dimensionless", 0.0 ); //for paraview visualization

	// [2026-09-18] Track the key phenotype-output nodes as custom_data on every cell,
	// regardless of which one is selected for coloring, so they are all available in the
	// saved output (MultiCellDS/legacy) for post-hoc analysis of the nutrient-release effect.
	for (std::string node : {"Apoptosis", "Proliferation", "Autophagy", "Senescence", "BECN1"})
	{
		if (node != parameters.strings("node_to_visualize"))
		{
			cell_defaults.custom_data.add_variable(node, "dimensionless", 0.0);
		}
	}

	// [2026-09-21] hard minimum-sustained-duration death commitment gate; see update_death_commitment().
	// apoptosis_on_since: PhysiCell_globals.current_time at which Apoptosis most recently turned
	// continuously ON, or -1.0 while Apoptosis is OFF. This is the source of truth -- it is a
	// timestamp difference against the simulation clock, not an accumulated per-tick dt, precisely
	// because the dt argument PhysiCell passes into pre_/post_update_intracellular is the BioFVM
	// diffusion_dt (confirmed by reading core/PhysiCell_cell_container.cpp), not the elapsed time
	// since the intracellular model last actually updated -- accumulating that dt would (and, in an
	// earlier version of this fix, silently did) undercount elapsed time by roughly
	// intracellular_dt/diffusion_dt, making the commit_duration threshold effectively unreachable.
	// apoptosis_sustained_min: derived, current-time-minus-onset value, recomputed each call purely
	// so it's visible/plottable in the saved output; not used as an accumulator.
	cell_defaults.custom_data.add_variable("apoptosis_on_since", "min", -1.0);
	cell_defaults.custom_data.add_variable("apoptosis_sustained_min", "min", 0.0);

	/*
	   This parses the cell definitions in the XML config file.
	*/
	
	initialize_cell_definitions_from_pugixml(); 
	
	/* 
	   Put any modifications to individual cell definitions here. 
	   
	   This is a good place to set custom functions. 
	*/ 
	
	/*
	   This builds the map of cell definitions and summarizes the setup. 
	*/

	build_cell_definitions_maps(); 

	/*
	   This intializes cell signal and response dictionaries 
	*/

	setup_signal_behavior_dictionaries();

	/*
	   This summarizes the setup. 
	*/
	
	display_cell_definitions( std::cout ); 


	return; 
}

void setup_microenvironment( void )
{
	// set domain parameters 
	
	// put any custom code to set non-homogeneous initial conditions or 
	// extra Dirichlet nodes here. 
	
	// initialize BioFVM 
	
	initialize_microenvironment(); 	
	
	return; 
}

void setup_tissue( void )
{
	// load cells from your CSV file
	load_cells_from_pugixml(); 	
}

void pre_update_intracellular( Cell* pCell, Phenotype& phenotype, double dt )
{
	// [2026-09-18] The original physiboss_cell_lines tutorial used this hook to ramp a
	// "$time_scale" MaBoSS parameter partway through the run; autophagy_network.cfg has no
	// such parameter, so there is nothing to do here for this project.
}

void post_update_intracellular( Cell* pCell, Phenotype& phenotype, double dt )
{
	update_death_commitment(pCell, dt);
	color_node(pCell);
}

void update_death_commitment( Cell* pCell, double dt )
{
	// [2026-09-21] Hard minimum-sustained-duration death commitment gate.
	//
	// Why this exists: PhysiBoSS's built-in output mapping turns a MaBoSS boolean node into a
	// continuous stochastic RATE each intracellular_dt (see addons/PhysiBoSS/src/maboss_intracellular.cpp,
	// update_outputs() -> PhysiCell::set_single_behavior()), smoothed/shaped by the <smoothing>/
	// <steepness> XML settings. That is fundamentally a *rate* mechanism: given enough time, any
	// sufficiently-sustained-but-not-permanent true signal can still push the cumulative death
	// probability arbitrarily close to 1, because the rate never resets to exactly zero between
	// two nearby true stretches. At max_time=1440 min this was tuned to work well (smoothing=30,
	// steepness=4, see DEV_LOG_2026-09-21.md), but at max_time=2880 min the same class of failure
	// re-appeared: a genuine, MaBoSS-confirmed ~270-consecutive-minute Apoptosis transient (which
	// resolves back to false and stays false for the rest of the run in the single-cell/notebook
	// trajectory) had enough cumulative time to also nearly-certainly trigger death under the rate
	// mapping, indistinguishable from true permanent commitment.
	//
	// Fix: bypass the rate mapping for Apoptosis entirely (removed from every config's
	// <mapping><output> block) and instead track, in custom_data, how many CONSECUTIVE minutes
	// Apoptosis has most recently been continuously ON, resetting to 0 the instant it goes false.
	// Only once that run length reaches apoptosis_commit_duration (a user parameter, default 600
	// min -- safely more than double the ~270 min longest observed transient, while still leaving
	// most of a 2880 min run for genuine commitments to be detected) do we deterministically call
	// Death::trigger_death(). This is a hard minimum-duration requirement, not a rate: a transient
	// that resolves before the threshold has zero chance of triggering death, no matter how many
	// times it recurs, and a cell that is genuinely permanently committed will always eventually
	// cross the threshold once and stay dead.
	//
	// Elapsed time is measured as a PhysiCell_globals.current_time timestamp difference, NOT by
	// accumulating the dt argument this function receives -- that dt is diffusion_dt (confirmed by
	// reading core/PhysiCell_cell_container.cpp: pre_/post_update_intracellular are invoked with
	// diffusion_dt_ every diffusion step, gated by need_update() so they only actually fire roughly
	// every intracellular_dt, but the value passed is still diffusion_dt_, not the elapsed time
	// since the previous call). Accumulating dt here undercounted elapsed time by roughly
	// intracellular_dt/diffusion_dt (about 50x in this project's configs), which silently made
	// apoptosis_commit_duration unreachable within any of these runs' max_time -- caught by
	// directly inspecting a single cell's saved apoptosis_on/sustained/dead trajectory after the
	// first full 12-condition verification run showed zero cells ever committing to death anywhere,
	// including the canonical amino-acid-starvation scenario where both the MaBoSS notebook and the
	// pre-fix PhysiBoSS model agree death should be universal.
	if ( pCell->phenotype.death.dead )
	{
		return;
	}

	bool apoptosis_on = pCell->phenotype.intracellular->get_boolean_variable_value( "Apoptosis" );

	if ( apoptosis_on )
	{
		if ( pCell->custom_data["apoptosis_on_since"] < 0.0 )
		{
			pCell->custom_data["apoptosis_on_since"] = PhysiCell_globals.current_time;
		}
		pCell->custom_data["apoptosis_sustained_min"] = PhysiCell_globals.current_time - pCell->custom_data["apoptosis_on_since"];
	}
	else
	{
		pCell->custom_data["apoptosis_on_since"] = -1.0;
		pCell->custom_data["apoptosis_sustained_min"] = 0.0;
	}

	static double commit_threshold = parameters.doubles("apoptosis_commit_duration");
	if ( pCell->custom_data["apoptosis_sustained_min"] >= commit_threshold )
	{
		static int necrosis_index = pCell->phenotype.death.find_death_model_index( "Necrosis" );

		// Death::trigger_death() by itself only sets the dead flag and the death-model index --
		// it does NOT switch the cycle model, stop motility/secretion, or run the death-phase
		// entry function (confirmed by reading its implementation in core/PhysiCell_phenotype.cpp,
		// where that block is commented out). Ordinarily that full transition happens once, right
		// after Death::check_for_death() returns true, in Cell::advance_bundled_phenotype_functions
		// (core/PhysiCell_cell.cpp). Since check_for_death() exits immediately once dead==true, it
		// will never run that block for a cell whose death we trigger here ourselves -- so we
		// replicate the same sequence PhysiCell's own core loop uses, line for line.
		Phenotype& p = pCell->phenotype;
		p.death.trigger_death( necrosis_index );
		p.cycle.sync_to_cycle_model( p.death.current_model() );

		p.motility.is_motile = false;
		p.motility.motility_vector.assign( 3, 0.0 );
		pCell->functions.update_migration_bias = NULL;

		p.secretion.set_all_secretion_to_zero();
		p.secretion.scale_all_uptake_by_factor( 0.10 );

		if ( p.cycle.current_phase().entry_function )
		{
			p.cycle.current_phase().entry_function( pCell, p, dt );
		}
	}
}

std::vector<std::string> my_coloring_function( Cell* pCell )
{
	std::vector< std::string > output( 4 , "rgb(0,0,0)" );

	// [2026-09-21] Multi-state coloring instead of a single node_to_visualize on/off split:
	// dying / senescent / autophagic / proliferating / other. Priority order handles a cell that
	// has more than one node true at once in the Boolean model (dying takes visual priority,
	// since it's usually the most clinically relevant state to spot). This does NOT read
	// node_to_visualize (User Params) at all -- that field only affects which node's value is
	// additionally tracked under a custom name in the saved output, not the on-screen color.
	// See legend.svg / README.md for the same color key.
	if ( pCell->phenotype.intracellular->get_boolean_variable_value( "Apoptosis" ) )
	{
		// dying -- red
		output[0] = "rgb(220,20,20)";
		output[2] = "rgb(120,0,0)";
	}
	else if ( pCell->phenotype.intracellular->get_boolean_variable_value( "Senescence" ) )
	{
		// senescent -- purple
		output[0] = "rgb(160,32,240)";
		output[2] = "rgb(90,0,140)";
	}
	else if ( pCell->phenotype.intracellular->get_boolean_variable_value( "Autophagy" ) )
	{
		// autophagic -- orange
		output[0] = "rgb(255,140,0)";
		output[2] = "rgb(160,80,0)";
	}
	else if ( pCell->phenotype.intracellular->get_boolean_variable_value( "Proliferation" ) )
	{
		// proliferating -- green
		output[0] = "rgb(0,180,0)";
		output[2] = "rgb(0,90,0)";
	}
	else
	{
		// none of the above (quiescent / undecided) -- gray
		output[0] = "rgb(170,170,170)";
		output[2] = "rgb(90,90,90)";
	}
	
	return output;
}

void color_node(Cell* pCell){
	std::string node_name = parameters.strings("node_to_visualize");
	pCell->custom_data[node_name] = pCell->phenotype.intracellular->get_boolean_variable_value(node_name);

	// [2026-09-18] also record the other tracked phenotype-output nodes (see create_cell_types)
	for (std::string node : {"Apoptosis", "Proliferation", "Autophagy", "Senescence", "BECN1"})
	{
		if (node != node_name)
		{
			pCell->custom_data[node] = pCell->phenotype.intracellular->get_boolean_variable_value(node);
		}
	}
}