
Element_target = {}
non_Element_target = {}
stimulation_check_number = 0
generated_file_path = "/plugins/stimulation/lua-stimulator-stim-codes.lua"

function initialize(box)
	dofile(box:get_config("${Path_Data}") .. generated_file_path)
	
	
	get_settings(box)

end

function get_settings(box)

	
	s_non_Element_target = box:get_setting(3)

	for i in s_non_Element_target:gmatch("%d+") do
		non_Element_target[i + 0] = true
	end
	
	s_Element_target = box:get_setting(2)
	for i in s_Element_target:gmatch("%d+") do
		Element_target[i + 0] = true
	end
	
	
	stimulation_check_number = _G[box:get_setting(4)]
	


end 


function uninitialize(box)
end

function process(box)

	is_finished = false

	while  not is_finished and box:keep_processing()  do

		current_time = box:get_current_time()

		while box:get_stimulation_count(1) > 0 do

			stimulation_code, stimulation_date, stimulation_duration = box:get_stimulation(1, 1)
			box:remove_stimulation(1, 1)
			
			if stimulation_code >= OVTK_StimulationId_Label_00 and stimulation_code <= OVTK_StimulationId_Label_1F then

				received_stimulation = stimulation_code - OVTK_StimulationId_Label_00

				if Element_target[received_stimulation] ~= nil then
					box:send_stimulation(1, stimulation_check_number, current_time)
				elseif non_Element_target[received_stimulation] ~= nil then
					box:send_stimulation(2, stimulation_check_number, current_time)
				end

			elseif stimulation_code == OVTK_StimulationId_ExperimentStop then
				is_finished = true
			end
		end

		box:sleep()

	end

end
