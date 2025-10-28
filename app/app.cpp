#include <imgui.h>
#include <backends/imgui_impl_sdl3.h>
#include <backends/imgui_impl_sdlgpu3.h>
#include <stdio.h>  // printf, fprintf
#include <stdlib.h> // abort
#include <SDL3/SDL.h>
#include <tinyfiledialogs.h>

#include <iostream>
#include <thread>
#include <exception>

// This example doesn't compile with Emscripten yet! Awaiting SDL3 support.
#ifdef __EMSCRIPTEN__
#include "../libs/emscripten/emscripten_mainloop_stub.h"
#endif

#include <spdlog/sinks/callback_sink.h>

#include <polyfem/State.hpp>
#include <polyfem/OptState.hpp>

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/JSONUtils.hpp>

using namespace polyfem;

bool load_json(const std::string &json_file, json &out)
{
	std::ifstream file(json_file);

	if (!file.is_open())
		return false;

	file >> out;

	if (!out.contains("root_path"))
		out["root_path"] = json_file;

	return true;
}

std::string openFileName(const std::string &defaultPath,
						 const std::vector<std::string> &filters, const std::string &desc)
{
	int n = filters.size();
	std::vector<char const *> filterPatterns(n);
	for (int i = 0; i < n; ++i)
	{
		filterPatterns[i] = filters[i].c_str();
	}
	char const *select = tinyfd_openFileDialog("Open File",
											   defaultPath.c_str(),
											   n,
											   filterPatterns.data(),
											   desc.c_str(), 0);
	if (select == nullptr)
	{
		return "";
	}
	else
	{
		return std::string(select);
	}
}

int run_polyfem(std::shared_ptr<State> state)
{
	try
	{
		std::vector<std::string> names;
		std::vector<Eigen::MatrixXi> cells;
		std::vector<Eigen::MatrixXd> vertices;

		state->load_mesh(/*non_conforming=*/false, names, cells, vertices);

		if (state->mesh == nullptr)
		{
			// Cannot proceed without a mesh.
			return EXIT_FAILURE;
		}

		state->stats.compute_mesh_stats(*state->mesh);

		state->build_basis();

		state->assemble_rhs();
		state->assemble_mass_mat();

		Eigen::MatrixXd sol;
		Eigen::MatrixXd pressure;

		state->solve_problem(sol, pressure);

		state->compute_errors(sol);

		logger().info("total time: {}s", state->timings.total_time());

		state->save_json(sol);
		state->export_data(sol, pressure);
		return EXIT_SUCCESS;
	}
	catch (const std::exception &e)
	{
		logger().error("Exception: {}", e.what());
		return EXIT_FAILURE;
	}
}

int run_opt(std::shared_ptr<OptState> opt_state)
{
	try
	{
		opt_state->create_states(opt_state->args["compute_objective"].get<bool>() ? polyfem::solver::CacheLevel::Solution : polyfem::solver::CacheLevel::Derivatives, opt_state->args["solver"]["max_threads"].get<int>());
		opt_state->init_variables();
		opt_state->create_problem();

		Eigen::VectorXd x;
		opt_state->initial_guess(x);

		if (opt_state->args["compute_objective"].get<bool>())
		{
			logger().info("Objective is {}", opt_state->eval(x));
			return EXIT_SUCCESS;
		}

		opt_state->solve(x);
		return EXIT_SUCCESS;
	}
	catch (const std::exception &e)
	{
		adjoint_logger().error("Exception: {}", e.what());
		return EXIT_FAILURE;
	}
}

void display_log(
	bool is_opt,
	const std::string &title,
	const std::vector<std::string> &nlogs,
	const std::vector<std::string> &adj_logs,
	int r, int g, int b)
{
	ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(r, g, b, 255));
	ImGui::Begin(title.c_str());
	ImGui::PopStyleColor();

	static bool show_opt_logs = false;

	if (is_opt)
	{
		if (ImGui::Button("Main"))
			show_opt_logs = false;
		ImGui::SameLine();
		if (ImGui::Button("Opt"))
			show_opt_logs = true;
	}

	const std::vector<std::string> logs =
		(is_opt && show_opt_logs) ? adj_logs : nlogs;
	ImGui::BeginChild("Scrolling");
	for (const auto &log : logs)
		ImGui::Text("%s", log.c_str());
	ImGui::EndChild();
	ImGui::End();
}

// Main code
int main(int argc, char **argv)
{
	if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD))
	{
		log_and_throw_error(fmt::format("Error: SDL_Init(): {}\n", SDL_GetError()));
		return EXIT_FAILURE;
	}

	// Create SDL window graphics context
	float main_scale = SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay());
	const int width = (int)(1280 * main_scale);
	const int height = (int)(720 * main_scale);

	SDL_WindowFlags window_flags = SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIDDEN | SDL_WINDOW_HIGH_PIXEL_DENSITY;
	SDL_Window *window = SDL_CreateWindow("PolyFEM", width, height, window_flags);
	if (window == nullptr)
	{
		log_and_throw_error(fmt::format("Error: SDL_CreateWindow(): {}\n", SDL_GetError()));
		return EXIT_FAILURE;
	}
	SDL_SetWindowPosition(window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
	SDL_ShowWindow(window);

	// Create GPU Device
	SDL_GPUDevice *gpu_device = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV | SDL_GPU_SHADERFORMAT_DXIL | SDL_GPU_SHADERFORMAT_METALLIB, true, nullptr);
	if (gpu_device == nullptr)
	{
		log_and_throw_error(fmt::format("Error: SDL_CreateGPUDevice(): {}\n", SDL_GetError()));
		return EXIT_FAILURE;
	}

	// Claim window for GPU Device
	if (!SDL_ClaimWindowForGPUDevice(gpu_device, window))
	{
		log_and_throw_error(fmt::format("Error: SDL_ClaimWindowForGPUDevice(): {}\n", SDL_GetError()));
		return EXIT_FAILURE;
	}
	SDL_SetGPUSwapchainParameters(gpu_device, window, SDL_GPU_SWAPCHAINCOMPOSITION_SDR, SDL_GPU_PRESENTMODE_MAILBOX);

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO &io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls

	// Setup Dear ImGui style
	// ImGui::StyleColorsDark();
	ImGui::StyleColorsLight();

	// Setup scaling
	ImGuiStyle &style = ImGui::GetStyle();
	style.ScaleAllSizes(main_scale); // Bake a fixed style scale. (until we have a solution for dynamic style scaling, changing this requires resetting Style + calling this again)
	// style.FontScaleDpi = main_scale; // Set initial font scale. (using io.ConfigDpiScaleFonts=true makes this unnecessary. We leave both here for documentation purpose)

	// Setup Platform/Renderer backends
	ImGui_ImplSDL3_InitForSDLGPU(window);
	ImGui_ImplSDLGPU3_InitInfo init_info = {};
	init_info.Device = gpu_device;
	init_info.ColorTargetFormat = SDL_GetGPUSwapchainTextureFormat(gpu_device, window);
	init_info.MSAASamples = SDL_GPU_SAMPLECOUNT_1;
	ImGui_ImplSDLGPU3_Init(&init_info);

	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

	std::vector<std::string> info;
	std::vector<std::string> warning;
	std::vector<std::string> debug;
	std::vector<std::string> error;
	std::vector<std::string> trace;

	std::vector<std::string> opt_info;
	std::vector<std::string> opt_warning;
	std::vector<std::string> opt_debug;
	std::vector<std::string> opt_error;
	std::vector<std::string> opt_trace;

	auto clear_logs = [&]() {
		info.clear();
		warning.clear();
		debug.clear();
		error.clear();
		trace.clear();

		opt_info.clear();
		opt_warning.clear();
		opt_debug.clear();
		opt_error.clear();
		opt_trace.clear();
	};

	auto callback_sink = std::make_shared<spdlog::sinks::callback_sink_mt>(
		[&info, &warning, &debug, &error, &trace](const spdlog::details::log_msg &msg) {
			time_t time = std::chrono::system_clock::to_time_t(msg.time);
			const auto str = fmt::format("[{}] {}",
										 std::put_time(std::localtime(&time), "%F %T"), msg.payload);
			switch (msg.level)
			{
			case spdlog::level::info:
				info.push_back(str);
				break;
			case spdlog::level::warn:
				warning.push_back(str);
				break;
			case spdlog::level::debug:
				debug.push_back(str);
				break;
			case spdlog::level::err:
				error.push_back(str);
				break;
			case spdlog::level::trace:
				trace.push_back(str);
				break;
			default:
				break;
			}
		});

	auto opt_callback_sink = std::make_shared<spdlog::sinks::callback_sink_mt>(
		[&opt_info, &opt_warning, &opt_debug, &opt_error, &opt_trace](const spdlog::details::log_msg &msg) {
			time_t time = std::chrono::system_clock::to_time_t(msg.time);
			const auto str = fmt::format("[{}] {}",
										 std::put_time(std::localtime(&time), "%F %T"), msg.payload);
			switch (msg.level)
			{
			case spdlog::level::info:
				opt_info.push_back(str);
				break;
			case spdlog::level::warn:
				opt_warning.push_back(str);
				break;
			case spdlog::level::debug:
				opt_debug.push_back(str);
				break;
			case spdlog::level::err:
				opt_error.push_back(str);
				break;
			case spdlog::level::trace:
				opt_trace.push_back(str);
				break;
			default:
				break;
			}
		});
	spdlog::logger logger("polyfem_app", {callback_sink});
	spdlog::logger opt_logger("polyfem_opt_app", {opt_callback_sink});

	set_logger(std::make_shared<spdlog::logger>(logger));
	set_adjoint_logger(std::make_shared<spdlog::logger>(opt_logger));

	std::shared_ptr<State> state;
	std::shared_ptr<OptState> opt_state;
	std::shared_ptr<std::thread> worker;

	// Main loop
	bool done = false;
	bool running = false;
	bool is_opt = false;
	bool json_loaded = false;
	std::string error_msg;
	std::string json_file;
	std::string output_dir;
	json in_args = json({});

	int t = 0, time_steps = 0;
	double tt = 0.0, tend = 0.0;

	static const char *log_levels[] = {
		"Trace",
		"Debug",
		"Info",
		"Warning",
		"Error",
		"Critical",
		"Off"};

	static int log_level = 1;
	static int opt_log_level = 1;

	while (!done)
	{
		SDL_Event event;
		while (SDL_PollEvent(&event))
		{
			ImGui_ImplSDL3_ProcessEvent(&event);
			if (event.type == SDL_EVENT_QUIT)
				done = true;
			if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED && event.window.windowID == SDL_GetWindowID(window))
				done = true;
		}

		if (SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED)
		{
			SDL_Delay(10);
			continue;
		}

		if (running && worker && !worker->joinable())
		{
			running = false;
			// is_opt = false;
		}

		// Start the Dear ImGui frame
		ImGui_ImplSDLGPU3_NewFrame();
		ImGui_ImplSDL3_NewFrame();
		ImGui::NewFrame();

		{
			ImGui::SetNextWindowSize(ImVec2(width, height / 4));
			ImGui::SetNextWindowPos(ImVec2(0, 0));
			ImGui::Begin("PolyFEM");
			{
				ImGui::BeginDisabled(running);

				if (ImGui::Button("Load JSON"))
				{
					clear_logs();
					json_loaded = false;
					json_file = openFileName("", {"*.json", "*.*"}, "JSON Files");
					if (!json_file.empty())
					{
						try
						{
							const bool ok = load_json(json_file, in_args);
							if (!ok)
							{
								error_msg = fmt::format("Failed to load JSON file: {}", json_file);
								std::cout << error_msg << std::endl;
								ImGui::OpenPopup("Error");
							}
							else
							{
								json_loaded = true;
								if (!in_args.contains("/output/directory"_json_pointer))
								{
									auto now = std::chrono::system_clock::now();
									auto duration = now.time_since_epoch();
									auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() / 10000;

									std::filesystem::path path(json_file);
									const auto tmp = path.parent_path() / "output" / fmt::format("{:04}", milliseconds);
									in_args["/output/directory"_json_pointer] = tmp.string();
								}
								output_dir = in_args["/output/directory"_json_pointer].get<std::string>();
							}
						}
						catch (const std::exception &e)
						{
							error_msg = fmt::format("Error loading JSON file: {}", e.what());
							std::cout << error_msg << std::endl;
							ImGui::OpenPopup("Error");
						}
					}
				}

				ImGui::SameLine();
				ImGui::SetNextItemWidth(100);
				ImGui::Combo("Main", &log_level, log_levels, IM_ARRAYSIZE(log_levels));
				ImGui::SameLine();
				ImGui::SetNextItemWidth(100);
				ImGui::Combo("Opt", &opt_log_level, log_levels, IM_ARRAYSIZE(log_levels));
				ImGui::SameLine();
				ImGui::BeginDisabled(!json_loaded);
				if (ImGui::Button("Run"))
				{
					running = false;
					is_opt = false;
					try
					{
						if (in_args.contains("states"))
						{
							opt_state = std::make_shared<OptState>();
							opt_state->init(in_args, false);

							// opt_state->opt_callback =
							// 	[&t, &time_steps, &tt, &tend](int tin, int time_stepsin, double ttin, double tendin) {
							// 		t = tin;
							// 		time_steps = time_stepsin;
							// 		tt = ttin;
							// 		tend = tendin;
							// 	};

							set_logger(std::make_shared<spdlog::logger>(logger));
							set_adjoint_logger(std::make_shared<spdlog::logger>(opt_logger));

							polyfem::logger().set_level(static_cast<spdlog::level::level_enum>(log_level));
							polyfem::adjoint_logger().set_level(static_cast<spdlog::level::level_enum>(log_level));

							running = true;
							is_opt = true;

							worker = std::make_shared<std::thread>(run_opt, opt_state);
							worker->detach();
						}
						else
						{

							state = std::make_shared<State>();
							state->init(in_args, true);

							state->time_callback =
								[&t, &time_steps, &tt, &tend](int tin, int time_stepsin, double ttin, double tendin) {
									t = tin;
									time_steps = time_stepsin;
									tt = ttin;
									tend = tendin;
								};
							set_logger(std::make_shared<spdlog::logger>(logger));
							polyfem::logger().set_level(static_cast<spdlog::level::level_enum>(log_level));
							running = true;

							worker = std::make_shared<std::thread>(run_polyfem, state);
							worker->detach();
						}
					}
					catch (const std::exception &e)
					{
						error_msg = fmt::format("{}", e.what());
						std::cout << error_msg << std::endl;
						ImGui::OpenPopup("Error");
					}
				}
				ImGui::EndDisabled();

				ImGui::Text("JSON File: %s", json_file.c_str());
				ImGui::Text("Output Directory: %s", output_dir.c_str());

				ImGui::EndDisabled();

				if (running)
				{
					ImGui::Text("Running...");
				}
				ImGui::ProgressBar(float(t) / float(time_steps), ImVec2(-1, 0), fmt::format("{}/{}, {:.2}s/{:.2}s", t, time_steps, tt, tend).c_str());
			}
			ImGui::End();

			if (ImGui::BeginPopupModal("Error"))
			{
				ImGui::Text("%s", error_msg.c_str());
				if (ImGui::Button("Close"))
					ImGui::CloseCurrentPopup();
				ImGui::EndPopup();
			}

			ImGui::SetNextWindowSize(ImVec2(width / 2, height * 3. / 8));
			ImGui::SetNextWindowPos(ImVec2(0, height / 4));
			display_log(is_opt, "Info", info, opt_info, 39, 174, 96);

			ImGui::SetNextWindowSize(ImVec2(width / 2, height * 3. / 8));
			ImGui::SetNextWindowPos(ImVec2(width / 2, height / 4));
			display_log(is_opt, "Warning", warning, opt_warning, 230, 126, 34);

			ImGui::SetNextWindowSize(ImVec2(width / 2, height * 3. / 8));
			ImGui::SetNextWindowPos(ImVec2(0, height / 4 + height * 3. / 8));
			display_log(is_opt, "Error", error, opt_error, 192, 57, 43);

			ImGui::SetNextWindowSize(ImVec2(width / 2, height * 3. / 16));
			ImGui::SetNextWindowPos(ImVec2(width / 2, height / 4 + height * 3. / 8));
			display_log(is_opt, "Debug", debug, opt_debug, 52, 152, 219);

			ImGui::SetNextWindowSize(ImVec2(width / 2, height * 3. / 16));
			ImGui::SetNextWindowPos(ImVec2(width / 2, height / 4 + height * 3. / 8 + height * 3. / 16));
			display_log(is_opt, "Trace", trace, opt_trace, 155, 89, 182);
		}

		// Rendering
		ImGui::Render();
		ImDrawData *draw_data = ImGui::GetDrawData();
		const bool is_minimized = (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f);

		SDL_GPUCommandBuffer *command_buffer = SDL_AcquireGPUCommandBuffer(gpu_device); // Acquire a GPU command buffer

		SDL_GPUTexture *swapchain_texture;
		SDL_AcquireGPUSwapchainTexture(command_buffer, window, &swapchain_texture, nullptr, nullptr); // Acquire a swapchain texture

		if (swapchain_texture != nullptr && !is_minimized)
		{
			// This is mandatory: call Imgui_ImplSDLGPU3_PrepareDrawData() to upload the vertex/index buffer!
			Imgui_ImplSDLGPU3_PrepareDrawData(draw_data, command_buffer);

			// Setup and start a render pass
			SDL_GPUColorTargetInfo target_info = {};
			target_info.texture = swapchain_texture;
			target_info.clear_color = SDL_FColor{clear_color.x, clear_color.y, clear_color.z, clear_color.w};
			target_info.load_op = SDL_GPU_LOADOP_CLEAR;
			target_info.store_op = SDL_GPU_STOREOP_STORE;
			target_info.mip_level = 0;
			target_info.layer_or_depth_plane = 0;
			target_info.cycle = false;
			SDL_GPURenderPass *render_pass = SDL_BeginGPURenderPass(command_buffer, &target_info, 1, nullptr);

			// Render ImGui
			ImGui_ImplSDLGPU3_RenderDrawData(draw_data, command_buffer, render_pass);

			SDL_EndGPURenderPass(render_pass);
		}

		// Submit the command buffer
		SDL_SubmitGPUCommandBuffer(command_buffer);
	}

	// Cleanup
	// [If using SDL_MAIN_USE_CALLBACKS: all code below would likely be your SDL_AppQuit() function]
	SDL_WaitForGPUIdle(gpu_device);
	ImGui_ImplSDL3_Shutdown();
	ImGui_ImplSDLGPU3_Shutdown();
	ImGui::DestroyContext();

	SDL_ReleaseWindowFromGPUDevice(gpu_device, window);
	SDL_DestroyGPUDevice(gpu_device);
	SDL_DestroyWindow(window);
	SDL_Quit();

	return 0;
}