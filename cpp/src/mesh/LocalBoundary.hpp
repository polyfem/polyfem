#ifndef LOCAL_BOUNDARY_HPP
#define LOCAL_BOUNDARY_HPP

#include <sstream>

namespace poly_fem
{
	class LocalBoundary
	{
	public:
		LocalBoundary()
		: flags_(0)
		{ }

		inline void set_left_boundary() { flags_ = flags_ | LEFT_MASK; }
		inline void set_right_boundary() { flags_ = flags_ | RIGHT_MASK; }

		inline void set_top_boundary() { flags_ = flags_ | TOP_MASK; }
		inline void set_bottom_boundary() { flags_ = flags_ | BOTTOM_MASK; }

		inline void set_front_boundary() { flags_ = flags_ | FRONT_MASK; }
		inline void set_back_boundary() { flags_ = flags_ | BACK_MASK; }


		inline void clear_left_boundary() { flags_ = flags_ & (63 ^ LEFT_MASK); }
		inline void clear_right_boundary() { flags_ = flags_ & (63 ^ RIGHT_MASK); }

		inline void clear_top_boundary() { flags_ = flags_ & (63 ^ TOP_MASK); }
		inline void clear_bottom_boundary() { flags_ = flags_ & (63 ^ BOTTOM_MASK); }

		inline void clear_front_boundary() { flags_ = flags_ & (63 ^ FRONT_MASK); }
		inline void clear_back_boundary() { flags_ = flags_ & (63 ^ BACK_MASK); }


		inline bool is_left_boundary() const { return (flags_ & LEFT_MASK) != 0; }
		inline bool is_right_boundary() const { return (flags_ & RIGHT_MASK) != 0; }

		inline bool is_top_boundary() const { return (flags_ & TOP_MASK) != 0; }
		inline bool is_bottom_boundary() const { return (flags_ & BOTTOM_MASK) != 0; }

		inline bool is_front_boundary() const { return (flags_ & FRONT_MASK) != 0; }
		inline bool is_back_boundary() const { return (flags_ & BACK_MASK) != 0; }

		inline bool is_boundary() const { return flags_ != 0; }

		inline std::string flags() const
		{
			std::stringstream ss;

			ss<<int(flags_)<<" ";
			ss<<(is_left_boundary()?"1":"0");
			ss<<(is_top_boundary()?"1":"0");
			ss<<(is_right_boundary()?"1":"0");
			ss<<(is_bottom_boundary()?"1":"0");

			return ss.str();
		}

		inline void set_left_edge_id(const int id) { edge_id_[0] = id; }
		inline void set_right_edge_id(const int id) { edge_id_[2] = id; }

		inline void set_top_edge_id(const int id) { edge_id_[1] = id; }
		inline void set_bottom_edge_id(const int id) { edge_id_[3] = id; }

		inline void set_front_edge_id(const int id) { edge_id_[4] = id; }
		inline void set_back_edge_id(const int id) { edge_id_[5] = id; }

		void clear_edge_tag(const int edge_id)
		{
			if(!is_boundary()) return;

			if(edge_id_[0] == edge_id) clear_left_boundary();
			if(edge_id_[1] == edge_id) clear_top_boundary();
			if(edge_id_[2] == edge_id) clear_right_boundary();
			if(edge_id_[3] == edge_id) clear_bottom_boundary();

			if(edge_id_[4] == edge_id) clear_front_boundary();
			if(edge_id_[5] == edge_id) clear_back_boundary();
		}

	private:
		char flags_;
		int edge_id_[6] = {-1, -1, -1, -1, -1, -1};

		static const int LEFT_MASK = 1;
		static const int TOP_MASK = 2;
		static const int RIGHT_MASK = 4;
		static const int BOTTOM_MASK = 8;
		static const int FRONT_MASK = 16;
		static const int BACK_MASK = 32;
	};
}

#endif //LOCAL_BOUNDARY_HPP
