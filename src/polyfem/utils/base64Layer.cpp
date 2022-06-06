#include "base64Layer.hpp"

namespace polyfem
{
	namespace utils
	{
		namespace
		{
			//! \cond fileScope
			//- The characters used for base-64
			static const unsigned char base64Chars[64] =
				{
					'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
					'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
					'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
					'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
					'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
					'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
					'w', 'x', 'y', 'z', '0', '1', '2', '3',
					'4', '5', '6', '7', '8', '9', '+', '/'};
			//! \endcond
		} // namespace

		std::size_t base64Layer::encodedLength(std::size_t n)
		{
			return 4 * ((n / 3) + (n % 3 ? 1 : 0));
		}

		inline unsigned char base64Layer::encode0() const
		{
			// Top 6 bits of char0
			return base64Chars[((group_[0] & 0xFC) >> 2)];
		}

		inline unsigned char base64Layer::encode1() const
		{
			// Bottom 2 bits of char0, Top 4 bits of char1
			return base64Chars[((group_[0] & 0x03) << 4) | ((group_[1] & 0xF0) >> 4)];
		}

		inline unsigned char base64Layer::encode2() const
		{
			// Bottom 4 bits of char1, Top 2 bits of char2
			return base64Chars[((group_[1] & 0x0F) << 2) | ((group_[2] & 0xC0) >> 6)];
		}

		inline unsigned char base64Layer::encode3() const
		{
			// Bottom 6 bits of char2
			return base64Chars[(group_[2] & 0x3F)];
		}

		void base64Layer::add(char c)
		{
			group_[groupLen_++] = static_cast<unsigned char>(c);
			if (groupLen_ == 3)
			{
				unsigned char out[4];

				out[0] = encode0();
				out[1] = encode1();
				out[2] = encode2();
				out[3] = encode3();
				os_.write(reinterpret_cast<char *>(out), 4);

				groupLen_ = 0;
			}

			dirty_ = true;
		}

		base64Layer::base64Layer(std::ostream &os)
			: os_(os),
			  group_(),
			  groupLen_(0),
			  dirty_(false)
		{
		}

		base64Layer::~base64Layer()
		{
			close();
		}

		void base64Layer::write(const char *s, std::streamsize n)
		{
			for (std::streamsize i = 0; i < n; ++i)
			{
				add(s[i]);
			}
		}

		void base64Layer::reset()
		{
			groupLen_ = 0;
			dirty_ = false;
		}

		bool base64Layer::close()
		{
			if (!dirty_)
			{
				return false;
			}

			unsigned char out[4];
			if (groupLen_ == 1)
			{
				group_[1] = 0;

				out[0] = encode0();
				out[1] = encode1();
				out[2] = '=';
				out[3] = '=';
				os_.write(reinterpret_cast<char *>(out), 4);
			}
			else if (groupLen_ == 2)
			{
				group_[2] = 0;

				out[0] = encode0();
				out[1] = encode1();
				out[2] = encode2();
				out[3] = '=';
				os_.write(reinterpret_cast<char *>(out), 4);
			}

			// group-length == 0 (no content)
			// group-length == 3 is not possible, already reset in add()

			groupLen_ = 0;
			dirty_ = false;

			return true;
		}
	} // namespace utils
} // namespace polyfem
