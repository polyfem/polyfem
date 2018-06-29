function(prepend_current_path SOURCE_FILES)
	# Use recursive substitution to expand SOURCE_FILES
	foreach(SOURCE_FILE ${${SOURCE_FILES}})
		set(MODIFIED ${MODIFIED} ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE})
	endforeach()
	set(${SOURCE_FILES} ${MODIFIED} PARENT_SCOPE)
endfunction()
