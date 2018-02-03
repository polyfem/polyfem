#pragma once

////////////////////////////////////////////////////////////////////////////////
#include <igl/viewer/Viewer.h>
#include <igl/viewer/ViewerPlugin.h>
////////////////////////////////////////////////////////////////////////////////

class ImGuiMenuBase : public igl::viewer::ViewerPlugin {
protected:
	// Hidpi scaling to be used for text rendering.
	float m_HidpiScaling;

	// Ratio between the framebuffer size and the window size.
	// May be different from the hipdi scaling!
	float m_PixelRatio;

public:
	virtual void init(igl::viewer::Viewer *_viewer) override;

	virtual void shutdown() override;

	virtual bool pre_draw() override;

	virtual bool post_draw() override;

	virtual void post_resize(int width, int height) override;

	// Mouse IO
	virtual bool mouse_down(int button, int modifier) override;

	virtual bool mouse_up(int button, int modifier) override;

	virtual bool mouse_move(int mouse_x, int mouse_y) override;

	virtual bool mouse_scroll(float delta_y) override;

	// Keyboard IO
	virtual bool key_pressed(unsigned int key, int modifiers) override;

	virtual bool key_down(int key, int modifiers) override;

	virtual bool key_up(int key, int modifiers) override;

	// Draw menu
	virtual void draw_menu();

	void draw_viewer_menu();

	void draw_labels_menu();

	void draw_labels();

	void draw_text(Eigen::Vector3d pos, Eigen::Vector3d normal, const std::string &text);

	float pixel_ratio();

	float hidpi_scaling();
};
