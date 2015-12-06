// ---------------------------------------------------------------------------------//
// Copyright (c) 2013, Regents of the University of Pennsylvania                    //
// All rights reserved.                                                             //
//                                                                                  //
// Redistribution and use in source and binary forms, with or without               //
// modification, are permitted provided that the following conditions are met:      //
//     * Redistributions of source code must retain the above copyright             //
//       notice, this list of conditions and the following disclaimer.              //
//     * Redistributions in binary form must reproduce the above copyright          //
//       notice, this list of conditions and the following disclaimer in the        //
//       documentation and/or other materials provided with the distribution.       //
//     * Neither the name of the <organization> nor the                             //
//       names of its contributors may be used to endorse or promote products       //
//       derived from this software without specific prior written permission.      //
//                                                                                  //
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND  //
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED    //
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE           //
// DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY               //
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES       //
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;     //
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND      //
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT       //
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS    //
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                     //
//                                                                                  //
// Contact Tiantian Liu (ltt1598@gmail.com) if you have any questions.              //
//----------------------------------------------------------------------------------//

#pragma warning( disable : 4996)

#include "anttweakbar_wrapper.h"
#include "mesh.h"
#include "simulation.h"

//----------Events Related Variables--------------------//
static int g_old_screen_width;
static int g_old_screen_height;
static ScalarType g_old_timestep;
static ScalarType g_old_stretch_stiffness;
static ScalarType g_old_bending_stiffness;
static ScalarType g_old_attachment_stiffness;

//----------Global Parameters----------------//
extern int g_screen_width;
extern int g_screen_height;

//----------State Control--------------------//
extern bool g_only_show_sim;
extern bool g_record;
extern bool g_pause;
extern bool g_show_wireframe;
extern bool g_show_texture;
extern bool g_GPU_render;
extern bool g_cloth_tear;
extern bool g_object_move;

//----------anttweakbar handlers----------//
extern void TW_CALL reset_simulation(void*);
extern void TW_CALL step_through(void*);

//----------key components--------------//
extern Mesh* g_mesh;
extern Simulation* g_simulation;

AntTweakBarWrapper::AntTweakBarWrapper()
{
}

AntTweakBarWrapper::~AntTweakBarWrapper()
{
    SaveSettings(NULL);
    Cleanup();
}

void AntTweakBarWrapper::Init()
{
    TwInit(TW_OPENGL, NULL);

    //Control Panel bar
    m_control_panel_bar = TwNewBar("Control Panel");
    TwDefine(" 'Control Panel' size='200 710' position='814 10' color='255 255 255' text=dark ");
    char control_bar_pos_string [255];
    sprintf(control_bar_pos_string, "'Control Panel' position='%d 10'", g_screen_width-210);
    TwDefine(control_bar_pos_string);
    // state control
    TwAddVarRW(m_control_panel_bar, "Pause", TwType(sizeof(bool)), &(g_pause), "group='State Control'");
    TwAddButton(m_control_panel_bar, "Step Once", step_through, NULL, "group='State Control' ");
    TwAddVarRW(m_control_panel_bar, "Record", TwType(sizeof(bool)), &(g_record), "group='State Control'");
	TwAddVarRW(m_control_panel_bar, "GPU", TwType(sizeof(bool)), &(g_GPU_render), "group='State Control'");
	TwAddVarRW(m_control_panel_bar, "Tear", TwType(sizeof(bool)), &(g_cloth_tear), "group='State Control'");
	TwAddVarRW(m_control_panel_bar, "Obj Move", TwType(sizeof(bool)), &(g_object_move), "group='State Control'");
    TwAddSeparator(m_control_panel_bar, NULL, "");
    // visualization
    TwAddVarRW(m_control_panel_bar, "Wireframe", TwType(sizeof(bool)), &(g_show_wireframe), "group='Visualization'");
    TwAddVarRW(m_control_panel_bar, "Texture", TwType(sizeof(bool)), &(g_show_texture), "group='Visualization'");
    TwAddVarRW(m_control_panel_bar, "Width", TW_TYPE_INT32, &(g_screen_width), "min=640 group='Screen Resolution'");
    TwAddVarRW(m_control_panel_bar, "Height", TW_TYPE_INT32, &(g_screen_height), "min=480 group='Screen Resolution'");
    TwAddSeparator(m_control_panel_bar, NULL, "");
    // buttons
    TwAddButton(m_control_panel_bar, "Save Settings", SaveSettings, this, " ");
    TwAddButton(m_control_panel_bar, "Load Settings", LoadSettings, this, " ");
    TwAddButton(m_control_panel_bar, "Default Settings", SetDefaultSettings, this, " ");
    TwAddSeparator(m_control_panel_bar, NULL, "");
    TwAddButton(m_control_panel_bar, "Reset Simulation", reset_simulation, NULL, " ");
    //!Control Panel bar

    // mesh settings bar
    m_mesh_bar = TwNewBar("Mesh Settings");
    TwDefine(" 'Mesh Settings' size='200 250' position='10 10' color='210 240 255' text=dark ");
    TwAddVarRW(m_mesh_bar, "Total Mass", TW_TYPE_SCALAR_TYPE, &(g_mesh->m_total_mass), " ");
    // tet settings
    TwAddVarRW(m_mesh_bar, "Tet File", TW_TYPE_CSSTRING(sizeof(g_mesh->m_tet_file_path)), &(g_mesh->m_tet_file_path), " group='Tet Settings' ");
    TwAddVarRW(m_mesh_bar, "Tet Scaling", TW_TYPE_SCALAR_TYPE, &(g_mesh->m_tet_scaling), " min=0.01 group='Tet Settings' ");
    // cloth settings
    // cloth dimensions
    TwAddVarRW(m_mesh_bar, "dim1", TW_TYPE_INT32, &(g_mesh->m_dim[0]), " label='Width' min=2 group='Cloth Dimension' ");
    TwAddVarRW(m_mesh_bar, "dim2", TW_TYPE_INT32, &(g_mesh->m_dim[1]), " label='Length' min=2 group='Cloth Dimension' ");
    TwDefine(" 'Mesh Settings'/'Cloth Dimension' group='Cloth Settings'");
    // cloth corner position1
    TwAddVarRW(m_mesh_bar, "c1x", TW_TYPE_SCALAR_TYPE, &(g_mesh->m_corners[0][0]), " label='X' group='Corner1 Position' ");
    TwAddVarRW(m_mesh_bar, "c1y", TW_TYPE_SCALAR_TYPE, &(g_mesh->m_corners[0][1]), " label='Y' group='Corner1 Position' ");
    TwAddVarRW(m_mesh_bar, "c1z", TW_TYPE_SCALAR_TYPE, &(g_mesh->m_corners[0][2]), " label='Z' group='Corner1 Position' ");
    TwDefine(" 'Mesh Settings'/'Corner1 Position' group='Cloth Settings'");
    //TwDefine(" 'Mesh Settings'/'Corner1 Position' opened=false ");
    // cloth corner position2
    TwAddVarRW(m_mesh_bar, "c2x", TW_TYPE_SCALAR_TYPE, &(g_mesh->m_corners[1][0]), " label='X' group='Corner2 Position' ");
    TwAddVarRW(m_mesh_bar, "c2y", TW_TYPE_SCALAR_TYPE, &(g_mesh->m_corners[1][1]), " label='Y' group='Corner2 Position' ");
    TwAddVarRW(m_mesh_bar, "c2z", TW_TYPE_SCALAR_TYPE, &(g_mesh->m_corners[1][2]), " label='Z' group='Corner2 Position' ");
    TwDefine(" 'Mesh Settings'/'Corner2 Position' group='Cloth Settings'");
    //TwDefine(" 'Mesh Settings'/'Corner2 Position' opened=false ");
    // !mesh settings bar

    // simulation settings bar
    m_sim_bar = TwNewBar("Simulation Settings");
    TwDefine(" 'Simulation Settings' size='200 450' position='10 270' color='255 216 224' text=dark ");
	// integration
    TwAddVarRW(m_sim_bar, "Time Step", TW_TYPE_SCALAR_TYPE, &g_simulation->m_h, " min=0.0001 step=0.0001 ");
	TwAddVarRW(m_sim_bar, "Iterations/Frame", TW_TYPE_INT32, &g_simulation->m_iterations_per_frame, " ");
    TwEnumVal integrationStyleEV[INTEGRATION_TOTAL_NUM] =  {{INTEGRATION_EXPLICIT_EULER, "Explicit Euler"},
                                                            {INTEGRATION_EXPLICIT_RK2, "Explicit RK2"},
                                                            {INTEGRATION_EXPLICIT_RK4, "Explicit RK4"},
                                                            {INTEGRATION_EXPLICIT_SYMPLECTIC, "Explicit Symplectic"},
                                                            {INTEGRATION_IMPLICIT_EULER_BARAFF_WITKIN, "Implicit Euler (BW)"},
                                                            {INTEGRATION_POSITION_BASED_DYNAMICS, "PBD"}};
    TwType integrationStyle = TwDefineEnum("Integration Method", integrationStyleEV, INTEGRATION_TOTAL_NUM);
    TwAddVarRW(m_sim_bar, "Method", integrationStyle, &g_simulation->m_integration_method, " ");
    // constants
	TwAddVarRW(m_sim_bar, "Attachment Stiffness", TW_TYPE_SCALAR_TYPE, &g_simulation->m_stiffness_attachment, " group='Spring Stiffness' ");
	TwAddVarRW(m_sim_bar, "Stretch Stiffness", TW_TYPE_SCALAR_TYPE, &g_simulation->m_stiffness_stretch, " group='Spring Stiffness' ");
	TwAddVarRW(m_sim_bar, "Bending Stiffness", TW_TYPE_SCALAR_TYPE, &g_simulation->m_stiffness_bending, " group='Spring Stiffness' ");
	TwDefine(" 'Simulation Settings'/'Spring Stiffness' group='Constants'");
	TwAddVarRW(m_sim_bar, "PBD Attachment Stiffness", TW_TYPE_SCALAR_TYPE, &g_simulation->m_stiffness_attachment_pbd, " group='PBD Stiffness' ");
	TwAddVarRW(m_sim_bar, "PBD Stretch Stiffness", TW_TYPE_SCALAR_TYPE, &g_simulation->m_stiffness_stretch_pbd, " group='PBD Stiffness' ");
	TwAddVarRW(m_sim_bar, "PBD Bending Stiffness", TW_TYPE_SCALAR_TYPE, &g_simulation->m_stiffness_bending_pbd, " group='PBD Stiffness' ");
	TwDefine(" 'Simulation Settings'/'PBD Stiffness' group='Constants'");
    TwAddVarRW(m_sim_bar, "Gravity", TW_TYPE_SCALAR_TYPE, &g_simulation->m_gravity_constant, " group='Constants' ");
    TwAddVarRW(m_sim_bar, "Damping Coefficient", TW_TYPE_SCALAR_TYPE, &g_simulation->m_damping_coefficient, " min=0 step=0.001 group='Constants' ");
	TwAddVarRW(m_sim_bar, "Restitution Coefficient", TW_TYPE_SCALAR_TYPE, &g_simulation->m_restitution_coefficient, "min=0 max=1 step=0.001 group='Constants' ");
	TwAddVarRW(m_sim_bar, "Cloth Tear Strength", TW_TYPE_SCALAR_TYPE, &g_simulation->m_cloth_tear_value, "min=1.2 max=5 step=0.01 group='Constants' ");
    // !simulation settings bar

    TwDefine(" TW_HELP visible=false ");
}

void AntTweakBarWrapper::Cleanup()
{
    m_control_panel_bar = NULL;
    m_mesh_bar = NULL;
    m_sim_bar = NULL;

    TwTerminate();
}

void AntTweakBarWrapper::Reset()
{
    Cleanup(); 
    Init();
}

void AntTweakBarWrapper::Hide()
{
    TwDefine(" 'Control Panel' visible=false ");
    TwDefine(" 'Mesh Settings' visible=false ");
    TwDefine(" 'Simulation Settings' visible=false ");
}

void AntTweakBarWrapper::Show()
{
    TwDefine(" 'Control Panel' visible=true ");
    TwDefine(" 'Mesh Settings' visible=true ");
    TwDefine(" 'Simulation Settings' visible=true ");
}

int AntTweakBarWrapper::Update()
{
    // update

    // control panel pos
    char control_bar_pos_string [255];
    sprintf(control_bar_pos_string, "'Control Panel' position='%d 10'", g_screen_width-210);
    TwDefine(control_bar_pos_string);

	if (g_simulation->m_integration_method == INTEGRATION_POSITION_BASED_DYNAMICS)
	{
		TwDefine(" 'Simulation Settings'/'PBD Stiffness' visible=true");
		TwDefine(" 'Simulation Settings'/'Spring Stiffness' visible=false");
	}
	else
	{
		TwDefine(" 'Simulation Settings'/'PBD Stiffness' visible=false");
		TwDefine(" 'Simulation Settings'/'Spring Stiffness' visible=true");
	}

    // mesh settings display
    switch (g_mesh->m_mesh_type)
    {
    case MESH_TYPE_TET:
        TwDefine(" 'Mesh Settings'/'Tet Settings' visible=true");
        TwDefine(" 'Mesh Settings'/'Cloth Settings' visible=false");
        break;
    case MESH_TYPE_CLOTH:
        TwDefine(" 'Mesh Settings'/'Tet Settings' visible=false");
        TwDefine(" 'Mesh Settings'/'Cloth Settings' visible=true");
        break;
    }

    // give feed back
    int atb_feedback = ATB_DEFAULT;
    if (g_old_screen_width!=g_screen_width || g_old_screen_height!=g_screen_height)
    {
        g_old_screen_width = g_screen_width;
        g_old_screen_height = g_screen_height;
        atb_feedback |= ATB_RESHAPE_WINDOW;
    }
    if (g_simulation->m_h != g_old_timestep)
    {
        g_old_timestep = g_simulation->m_h;

        atb_feedback |= ATB_CHANGE_TIME_STEP;
    }
    if (g_old_stretch_stiffness != g_simulation->m_stiffness_stretch ||\
        g_old_bending_stiffness != g_simulation->m_stiffness_bending ||\
        g_old_attachment_stiffness != g_simulation->m_stiffness_attachment)
    {
        g_old_stretch_stiffness = g_simulation->m_stiffness_stretch;
        g_old_bending_stiffness = g_simulation->m_stiffness_bending;
        g_old_attachment_stiffness = g_simulation->m_stiffness_attachment;

        atb_feedback |= ATB_CHANGE_STIFFNESS;
    }
    
    return atb_feedback;
}

void AntTweakBarWrapper::SaveSettings()
{
    std::ofstream outfile;
    outfile.open(DEFAULT_CONFIG_FILE, std::ifstream::out);
    if (outfile.is_open())
    {
        // TODO: change it to memory dump.
        // global settings:
        outfile << "Wireframe              " << g_show_wireframe << std::endl;
        outfile << "Texture                " << g_show_texture << std::endl;
        outfile << "ScreenWidth            " << g_screen_width << std::endl;
        outfile << "ScreenHeight           " << g_screen_height << std::endl;
        outfile << std::endl;

        // mesh settings:
        outfile << "MeshType               " << g_mesh->m_mesh_type << std::endl;
        outfile << "MeshMass               " << g_mesh->m_total_mass << std::endl;
        outfile << "TetFilePath            " << g_mesh->m_tet_file_path<< std::endl;
        outfile << "TetScaling             " << g_mesh->m_tet_scaling << std::endl;
        outfile << "ClothDimension         " << g_mesh->m_dim[0] << " " \
                                             << g_mesh->m_dim[1] \
                                             << std::endl;
        outfile << "ClothCorners           " << g_mesh->m_corners[0][0] << " " \
                                             << g_mesh->m_corners[0][1] << " " \
                                             << g_mesh->m_corners[0][2] << " " \
                                             << g_mesh->m_corners[1][0] << " " \
                                             << g_mesh->m_corners[1][1] << " " \
                                             << g_mesh->m_corners[1][2] \
                                             << std::endl;
        outfile << std::endl;

        // simulation settings:
        outfile << "SimMethod              " << g_simulation->m_integration_method << std::endl;
        outfile << "Timestep               " << g_simulation->m_h << std::endl;

		outfile << "AttachmentStiffness    " << g_simulation->m_stiffness_attachment << std::endl;
		outfile << "StretchStiffness       " << g_simulation->m_stiffness_stretch << std::endl;
		outfile << "BendingStiffness       " << g_simulation->m_stiffness_bending << std::endl;
		outfile << "AttachmentStiffnessPBD " << g_simulation->m_stiffness_attachment_pbd << std::endl;
		outfile << "StretchStiffnessPBD    " << g_simulation->m_stiffness_stretch_pbd << std::endl;
		outfile << "BendingStiffnessPBD    " << g_simulation->m_stiffness_bending_pbd << std::endl;
        outfile << "GravityConstant        " << g_simulation->m_gravity_constant << std::endl;
		outfile << "DampingCoefficient     " << g_simulation->m_damping_coefficient << std::endl;
		outfile << "RestitutionCoefficient " << g_simulation->m_restitution_coefficient << std::endl;

        outfile << "IterationsPerFrame     " << g_simulation->m_iterations_per_frame << std::endl;

        outfile.close();
    }
    else
    {
        std::cerr << "Warning: Can not write config file. Settings not saved." << std::endl; 
    }
}

void AntTweakBarWrapper::LoadSettings()
{
    bool successfulRead = false;

    //read file
    std::ifstream infile;
    infile.open(DEFAULT_CONFIG_FILE, std::ifstream::in);
    if (successfulRead = infile.is_open())
    {
        int tempEnum;
        char ignoreToken[256];

        // global settings:
        infile >> ignoreToken >> g_show_wireframe;
        infile >> ignoreToken >> g_show_texture;
        infile >> ignoreToken >> g_screen_width;
        infile >> ignoreToken >> g_screen_height;

        // mesh settings:
        infile >> ignoreToken >> tempEnum; g_mesh->m_mesh_type = MeshType(tempEnum);
        infile >> ignoreToken >> g_mesh->m_total_mass;
        infile >> ignoreToken >> g_mesh->m_tet_file_path;
        infile >> ignoreToken >> g_mesh->m_tet_scaling;
        infile >> ignoreToken >> g_mesh->m_dim[0] \
                              >> g_mesh->m_dim[1];
        infile >> ignoreToken >> g_mesh->m_corners[0][0] \
                              >> g_mesh->m_corners[0][1] \
                              >> g_mesh->m_corners[0][2] \
                              >> g_mesh->m_corners[1][0] \
                              >> g_mesh->m_corners[1][1] \
                              >> g_mesh->m_corners[1][2];                                         ;

        // simulation settings:
        infile >> ignoreToken >> tempEnum; g_simulation->m_integration_method = IntegrationMethod(tempEnum);
        infile >> ignoreToken >> g_simulation->m_h;

		infile >> ignoreToken >> g_simulation->m_stiffness_attachment;
		infile >> ignoreToken >> g_simulation->m_stiffness_stretch;
		infile >> ignoreToken >> g_simulation->m_stiffness_bending;
		infile >> ignoreToken >> g_simulation->m_stiffness_attachment_pbd;
		infile >> ignoreToken >> g_simulation->m_stiffness_stretch_pbd;
		infile >> ignoreToken >> g_simulation->m_stiffness_bending_pbd;
        infile >> ignoreToken >> g_simulation->m_gravity_constant;
        infile >> ignoreToken >> g_simulation->m_damping_coefficient;
		infile >> ignoreToken >> g_simulation->m_restitution_coefficient;

        infile >> ignoreToken >> g_simulation->m_iterations_per_frame;

        infile.close();
    }

    // setup default values
    if (!successfulRead)
    {
        std::cerr << "Waning: failed loading settings, set to defaults." << std::endl;
        DefaultSettings();
    }

    // init event related variables
    g_old_screen_width = g_screen_width;
    g_old_screen_height = g_screen_height;
}

void AntTweakBarWrapper::DefaultSettings()
{
    // global settings
    g_show_wireframe = false;
    g_show_texture = false;
    g_screen_width = 1024;
    g_screen_height = 768;

    // mesh settings
    g_mesh->m_mesh_type = MESH_TYPE_CLOTH;
    g_mesh->m_total_mass = 1.0;
    // tet
    strcpy(g_mesh->m_tet_file_path, DEFAULT_MODEL);
    g_mesh->m_tet_scaling = 1.0;
    // cloth
    g_mesh->m_dim[0] = 21;
    g_mesh->m_dim[1] = 21;
    g_mesh->m_corners[0] = EigenVector3(-5, 8, -5); 
    g_mesh->m_corners[1] = EigenVector3(5, -1.85, -3.26);

    //simulation settings
    g_simulation->m_integration_method = INTEGRATION_EXPLICIT_EULER;
    g_simulation->m_h = 0.0333;

	g_simulation->m_stiffness_attachment = 120;
	g_simulation->m_stiffness_stretch = 80;
	g_simulation->m_stiffness_bending = 20;
	g_simulation->m_stiffness_attachment_pbd = 1;
	g_simulation->m_stiffness_stretch_pbd = 0.9;
	g_simulation->m_stiffness_bending_pbd = 0.9;
    g_simulation->m_gravity_constant = 100;
	g_simulation->m_damping_coefficient = 0.001;
	g_simulation->m_restitution_coefficient = 0.1;

    g_simulation->m_iterations_per_frame = 10;
}

void TW_CALL AntTweakBarWrapper::SaveSettings(void* atb_wrapper)
{
    AntTweakBarWrapper* atb_wrapper_ref = (AntTweakBarWrapper*) atb_wrapper;
    atb_wrapper_ref->SaveSettings();
}

void TW_CALL AntTweakBarWrapper::LoadSettings(void* atb_wrapper)
{
    AntTweakBarWrapper* atb_wrapper_ref = (AntTweakBarWrapper*) atb_wrapper;
    atb_wrapper_ref->LoadSettings();
    //resetSimulation(NULL);
}

void TW_CALL AntTweakBarWrapper::SetDefaultSettings(void* atb_wrapper)
{
    AntTweakBarWrapper* atb_wrapper_ref = (AntTweakBarWrapper*) atb_wrapper;
    atb_wrapper_ref->DefaultSettings();
    //resetSimulation(NULL);
}