
S_freq = {}
Frequency_NUM = 0

Target_L_Color = {}
Target_D_Color = {}
target_training_Cap = {}
target_training_pos = {}

epoch_time = nil
epoc_delay = nil
freq_tolerance = nil
CONSTANT_LUA_GEN =  "/plugins/stimulation/lua-stimulator-stim-codes.lua"

Chs = nil

function initialize(box)


	
	dofile(box:get_config("${Path_Data}") ..CONSTANT_LUA_GEN)

	Get_settings(box)

	

end

function uninitialize(box)
end

--Main process function
function process(box)

	while  box:get_stimulation_count(1) == 0 and box:keep_processing() do
		box:sleep()
	end

	box:log("Info", box:get_config("Running Configuration for the experiment"))
	box:log("Info", box:get_config("Writing..... to  '${CustomConfigurationPrefix${OperatingSystem}}-ssvep-demo${CustomConfigurationSuffix${OperatingSystem}}'"))

	if (Write_Config_to_file(box) == false) then return false end
	if (Temporal_config(box) == false)  then return false end
	if (Time_based_epoch(box) == false) then return false end
	if(Channel_selector(box) == false) then return false end
	
	

	
	
	-- notify the scenario that the configuration process is complete
	box:send_stimulation(1, OVTK_StimulationId_TrainCompleted, box:get_current_time() + 0.2, 0)

end

-- Get the Values from the table and input --
function Get_settings(box)

	epoch_time = box:get_setting(5)
	epoc_delay = box:get_setting(6)
	freq_tolerance = box:get_setting(7)
	Chs  = box:get_setting(8)
	
	for i in box:get_setting(4):gmatch("%d+[.]?%d*") do
		table.insert(S_freq, i)
		Frequency_NUM = Frequency_NUM + 1
	end
		for i in box:get_setting(3):gmatch("%d+") do
		table.insert(Target_D_Color, i)
	end
	for i in box:get_setting(2):gmatch("%d+") do
		table.insert(Target_L_Color, i)
	end
	

end

-- Write to exp config file
function Write_Config_to_file(box)
	myFile = assert(io.open(box:get_config("${CustomConfigurationPrefix${OperatingSystem}}-ssvep-demo${CustomConfigurationSuffix${OperatingSystem}}"), "a"))
	
	success = true 
	success = success and myFile:write("SSVEP_TargetLightColourRed = ", Target_L_Color[1] / 100, "\n")
	success = success and myFile:write("SSVEP_TargetLightColourGreen = ", Target_L_Color[2] / 100, "\n")
	success = success and myFile:write("SSVEP_TargetLightColourBlue = ", Target_L_Color[3] / 100, "\n")
	success = success and myFile:write("SSVEP_TargetDarkColourRed = ", Target_D_Color[1] / 100, "\n")
	success = success and myFile:write("SSVEP_TargetDarkColourGreen = ", Target_D_Color[2] / 100, "\n")
	success = success and myFile:write("SSVEP_TargetDarkColourBlue = ", Target_D_Color[3] / 100, "\n")

	for i=1,Frequency_NUM do
		success = success and myFile:write("SSVEP_Frequency_", i, " = ", string.format("%g", S_freq[i]), "\n")
	end
	
	myFile:close()

	if (success == false) then
		box:log("Error", box:get_config("Couldnt Write  '${CustomConfigurationPrefix${OperatingSystem}}-ssvep-demo${CustomConfigurationSuffix${OperatingSystem}}'  config file" ))
		return false
	else  return true 
	end
	
end 

-- create config files for temporal filters
function Temporal_config(box)
	
	
	

	scenario_path = box:get_config("${Player_ScenarioDirectory}")
	
	box:log("Info", "Frequency Count : '" ..Frequency_NUM.. "'")
	
	for i=1,Frequency_NUM do
		myFile_name = scenario_path .. string.format("/configuration/temporal-filter-freq-%d.cfg", i)
		
		myFile = io.open(myFile_name, "w")
		
		if myFile == nil then
			box:log("Error", "Cannot write to [" .. myFile_name .. "]")	
			return false
		end
			box:log("Info", "Stimulation Frequency: '"..S_freq[i].."'")
	box:log("Info", " tolerance against the Stimulation freq =  '"..tostring(S_freq[i] - freq_tolerance).."'")
		success = true
		success = success and myFile:write("<OpenViBE-SettingsOverride>\n")
		success = success and myFile:write("<SettingValue>Butterworth</SettingValue>\n")
		success = success and myFile:write("<SettingValue>Band pass</SettingValue>\n")
		success = success and myFile:write("<SettingValue>4</SettingValue>\n")
		success = success and myFile:write(string.format("<SettingValue>%g</SettingValue>\n", tostring(S_freq[i] - freq_tolerance)))
		success = success and myFile:write(string.format("<SettingValue>%g</SettingValue>\n", tostring(S_freq[i] + freq_tolerance)))
		success = success and myFile:write("<SettingValue>0.500000</SettingValue>\n")
		success = success and myFile:write("</OpenViBE-SettingsOverride>\n")
		
		myFile:close()
		
		if (success == false) then
			box:log("Error", box:get_config("Write error"))
			return false
		end
	
	myfilename = "${Player_ScenarioDirectory}/configuration/temporal-filter-freq-%dh1.cfg"
	myFile = assert(io.open(string.format(box:get_config( myfilename) , i ), "w"))
	
		myFile:write("<OpenViBE-SettingsOverride>\n")
		myFile:write("<SettingValue>Butterworth</SettingValue>\n")
		myFile:write("<SettingValue>Band pass</SettingValue>\n")
		myFile:write("<SettingValue>4</SettingValue>\n")
		myFile:write(string.format("<SettingValue>%s</SettingValue>\n", tostring(S_freq[i] * 2 - freq_tolerance)))
		myFile:write(string.format("<SettingValue>%s</SettingValue>\n", tostring(S_freq[i] * 2 + freq_tolerance)))
		myFile:write("<SettingValue>0.500000</SettingValue>\n")
		myFile:write("</OpenViBE-SettingsOverride>\n")

		myFile:close()
		
		box:log("Info", "Writing into the file [" ..myfilename.."]")
	
	end 
	
end

-- create configuration file for time based epoching--

function Time_based_epoch(box)


	myFile_name = scenario_path .. "/configuration/time-based-epoching.cfg";
	
	box:log("Info", "Writing file '" .. myFile_name .. "'")

	myFile = io.open(myFile_name, "w")
	if myFile == nil then
		box:log("Error", "Cannot write to [" .. myFile_name .. "]")
		return false
	end
		
	success = true
	success = success and myFile:write("<OpenViBE-SettingsOverride>\n")
	success = success and myFile:write(string.format("<SettingValue>%g</SettingValue>\n", tostring(epoch_time)))
	success = success and myFile:write(string.format("<SettingValue>%g</SettingValue>\n", tostring(epoc_delay)))
	success = success and myFile:write("</OpenViBE-SettingsOverride>\n")
		
	myFile:close()

	if (success == false) then
		box:log("Error", box:get_config("Write error"))
		return false
	else  return true end 
end
	
function Channel_selector(box)
	
	myfilename = scenario_path .."/configuration/channel-selector.cfg"
		box:log("Info", "Writing file '" .. myfilename .. "'")
	
	myFile = io.open(myfilename, "w")
	
	success = true
	
	success = success and myFile:write("<OpenViBE-SettingsOverride>\n")
	success = success and myFile:write(string.format("<SettingValue>%s</SettingValue>\n", Chs))
	success = success and myFile:write(string.format("<SettingValue>%s</SettingValue>\n", "Select"))
	success = success and myFile:write(string.format("<SettingValue>%s</SettingValue>\n", "Smart"))
	success = success and myFile:write("</OpenViBE-SettingsOverride>\n")
	
	myFile:close()
	
	if(success == false) then 
	box:log("Error", "Writing into File ["..myfilename.."]" )
	
	return false 
	else return true 
	end 
	
	
	

end