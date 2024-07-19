
f_count = 0
switched_count = 0
f_list = {}


Generated_file_path = "/plugins/stimulation/lua-stimulator-stim-codes.lua"
function initialize(box)
	dofile(box:get_config("${Path_Data}") .. Generated_file_path)

	f_count = box:get_input_count()

	for i = 1, f_count do
		f_list[i] = false
	end
end

function uninitialize(box)
end


function write_flip(i, j)

	io.write("Flip ", i, " of ", j, " switched\n")

end 
function stimulation_alter(box)


	while box:keep_processing() and switched_count < f_count do
		for i = 1, f_count do
		
			if box:get_stimulation_count(i) > 0 then 
		
				box:remove_stimulation(i,1)
			
				if not f_list[i] then 
			
					f_list[i] = true
					switched_count = switched_count +1
				
					write_flip(i,f_count)
				
				end 
			end 
		end 
		box:sleep()
		
	end 
end
function process(box)

	stimulation_alter(box)
	box:send_stimulation(1, OVTK_StimulationId_Label_00, box:get_current_time())
end
