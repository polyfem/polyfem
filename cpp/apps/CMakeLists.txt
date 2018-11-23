cmake_minimum_required(VERSION 3.1)


polyfem_add_application(bc_setter.cpp polyfem igl::core igl::tetgen geogram cli11)
polyfem_add_application(b_extractor.cpp polyfem igl::core geogram cli11)

polyfem_add_application(extrude_mesh.cpp polyfem cli11)
polyfem_add_application(square_beam.cpp polyfem cli11)
polyfem_add_application(tetrahedralize.cpp igl::core igl::tetgen geogram)
polyfem_add_application(tetrahedralize_hybrid.cpp polyfem igl::core igl::tetgen geogram cli11)



set_target_properties(bc_setter PROPERTIES FOLDER app)
set_target_properties(b_extractor PROPERTIES FOLDER app)

set_target_properties(extrude_mesh PROPERTIES FOLDER app)
set_target_properties(square_beam PROPERTIES FOLDER app)
set_target_properties(tetrahedralize PROPERTIES FOLDER app)
set_target_properties(tetrahedralize_hybrid PROPERTIES FOLDER app)

