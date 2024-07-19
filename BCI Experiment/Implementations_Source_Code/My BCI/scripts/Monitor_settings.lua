
vrpn_host = nil
vrpn_port = nil
--Screen Referesh rate--
Screen_RF = nil

file_dir = "${CustomConfigurationPrefix${OperatingSystem}}-ssvep-demo${CustomConfigurationSuffix${OperatingSystem}}"
CONSTANT_LUA_GEN =  "/plugins/stimulation/lua-stimulator-stim-codes.lua"



function initialize(box)
	dofile(box:get_config("${Path_Data}") ..CONSTANT_LUA_GEN)
	Screen_RF = box:get_setting(2)
end

function uninitialize(box)
end

function process(box)

	box:log("Info", box:get_config("Generating ".. file_dir))

	myfile = assert(io.open(box:get_config(file_dir), "w"))
	
	
	
	if (loadFile(myfile,box) == false) then
		
		box:log("Error", box:get_config("Couldnt Write"))
	end
	
	myfile:close()

	box:send_stimulation(1, OVTK_StimulationId_TrainCompleted, box:get_current_time() + 0.2, 0)
end




function loadFile(myfile,box)


	box:log("Info", "Attempting to write in the SSVEP_ScreenRefreshRate.config")
	success = true
	success = success and myfile:write("SSVEP_ScreenRefreshRate = ", Screen_RF, "\n")

	if(success == false) then
	
		return false
	
	else 
		return true
	
	end 

end 