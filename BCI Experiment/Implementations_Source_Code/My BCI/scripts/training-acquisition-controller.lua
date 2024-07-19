-- training aquisition controller
Sq = {}
cycles_num = 0

stim_time = nil
break_time = nil
flashing_rate = nil

t_width = nil
t_height = nil

t_pos = {}
t_nums = {}

stimulationLabels = {
	0x00008100,
	0x00008101,
	0x00008102,
	0x00008103,
	0x00008104,
	0x00008105,
	0x00008106,
	0x00008107
}

function process_settings(box)

	for t in s_Sq:gmatch("%d+") do	
		
		table.insert(Sq, t)
		cycles_num = cycles_num +1
		
	end 

	if (s_break_time:find("^%d+[.]?%d*$") ~= nil) then
		break_time = tonumber(s_break_time)
		box:log("Info", string.format(">>>Break time or durtation : [%s]", s_break_time))
	else
		box:log("Error", "problem with the parameter in the break duration \n")
		error()
	end
	if (s_flashing_rate:find("^%d+[.]?%d*$") ~= nil) then
		flashing_rate = tonumber(s_flashing_rate)
		box:log("Info", string.format("Flashing rate : [%s]", s_flashing_rate))
	else
		box:log("Error", "problem with the parameter in the Flashing rate \n")
		error()
	end
	
	if (s_stim_time:find("^%d+[.]?%d*$") ~= nil) then
		stim_time = tonumber(s_stim_time)
		box:log("Info", string.format(">>>Stimulation time : [%g]", stim_time))
	else
		box:log("Error", "problem with the parameter in the stimulation duration \n")
		error()
	end
	if s_width ~= nil and s_height ~= nil then
		box:log("Info", string.format("Target shape : width = %g, height = %g", t_width, t_height))
	else
		box:log("Error", "problem with the parameter in the Traget Shape \n")
		error()
	end
	
	t_nums = 0

	for s_target_x, s_target_y in s_targetPositions:gmatch("(-?%d+[.]?%d*);(-?%d+[.]?%d*)") do
		box:log("Info", string.format("Target %d : x = %g y = %g", t_nums, tonumber(s_target_x), tonumber(s_target_y)))
		table.insert(t_pos, {tonumber(s_target_x), tonumber(s_target_y)})
		t_nums = t_nums + 1

	end

end
function load_settings(box)
	
	-- load the goal Sq
	s_Sq = box:get_setting(2)
	-- get the duration of a stimulation Sq
	s_stim_time = box:get_setting(3)
	-- get the duration of a break between stimulations
	s_break_time = box:get_setting(4)
	-- get the delay between the appearance of the marker and the start of flickering
	s_flashing_rate = box:get_setting(5)
	-- get the target size
	s_targetSize = box:get_setting(6)
	-- get the targets' positions
	s_targetPositions = box:get_setting(7)
	


	s_width, s_height = s_targetSize:match("^(%d+[.]?%d*);(%d+[.]?%d*)$")
	t_width = tonumber(s_width)
	t_height = tonumber(s_height)

	
	


end 

-- create the configuration file for the stimulation-based-epoching
--this file is used during classifier training only
function config_maker_sbp(box)
	
	myfile_name = box:get_config("${Player_ScenarioDirectory}/configuration/stimulation-based-epoching.cfg")
	myFile = io.open(myfile_name, "w")
	box:log("Info", "Writing into [" .. myfile_name .. "]")
	if myFile == nil then 
		box:log("Error", "Cannot write to [" .. myfile_name .. "]")
		return false
	end 
	
	myFile:write("<OpenViBE-SettingsOverride>\n")
	myFile:write("	<SettingValue>", stim_time, "</SettingValue>\n")
	myFile:write("	<SettingValue>", flashing_rate, "</SettingValue>\n")
	myFile:write("	<SettingValue>OVTK_StimulationId_Target</SettingValue>\n")
	myFile:write("</OpenViBE-SettingsOverride>\n")
	myFile:close()


end
-- create the configuration file for the training program
function config_maker_tp(box)

	file_name = "${CustomConfigurationPrefix${OperatingSystem}}-ssvep-demo-training${CustomConfigurationSuffix${OperatingSystem}}"
	
	myFile = io.open(box:get_config(file_name),"w")
	box:log("Info","Writing into ["..file_name.."]")
	
	if myFile == nil then 
		box:log("Error", "Cannot write to [".. file_name.."]")
		return false 
	end 

	myFile:write("SSVEP_TargetCount = ", t_nums, "\n")
	myFile:write("SSVEP_TargetWidth = ", t_width, "\n")
	myFile:write("SSVEP_TargetHeight = ", t_height, "\n")
	
	for target_index, position in ipairs(t_pos) do

		myFile:write("SSVEP_Target_X_", target_index - 1, " = ", position[1], "\n")
		myFile:write("SSVEP_Target_Y_", target_index - 1, " = ", position[2], "\n")
	end

	myFile:close()

end 
function initialize(box)


	dofile(box:get_config("${Path_Data}") .. "/plugins/stimulation/lua-stimulator-stim-codes.lua")

	load_settings(box)
	
	process_settings(box)

	if(config_maker_sbp(box) == false) then  return false end 
	if (config_maker_tp(box) == false) then return false end 
	
	

end

function uninitialize(box)
end

function process(box)

	while box:keep_processing() and box:get_stimulation_count(1) == 0 do
		box:sleep()
	end

	current_time = box:get_current_time() + 1

	box:send_stimulation(1, OVTK_StimulationId_ExperimentStart, current_time, 0)

	current_time = current_time + 2

	for i,j in ipairs(Sq) do
		box:log("Info", string.format("Goal no %d is %d at %d", i, j, current_time))
		-- mark goal
		box:send_stimulation(2, OVTK_StimulationId_LabelStart + j, current_time, 0)
		-- wait for flashing_rate seconds
		current_time = current_time + flashing_rate
		-- start flickering
		box:send_stimulation(1, OVTK_StimulationId_VisualStimulationStart, current_time, 0)
		-- wait for stim_time seconds
		current_time = current_time + stim_time
		-- unmark goal and stop flickering
		box:send_stimulation(1, OVTK_StimulationId_VisualStimulationStop, current_time, 0)
		-- wait for break_time seconds
		current_time = current_time + break_time
	end

	box:send_stimulation(1, OVTK_StimulationId_ExperimentStop, current_time, 0)

	box:sleep()
end

