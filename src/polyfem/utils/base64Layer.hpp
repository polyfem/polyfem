#pragma once

#include <iostream>

namespace polyfem
{
	namespace utils
	{
		class base64Layer
		{
		private:
			//- The output stream for the layer
			std::ostream &os_;

			//- Buffer of characters to encode
			unsigned char group_[3];

			//- Current length of the encode buffer
			unsigned char groupLen_;

			//- Track if anything has been encoded.
			bool dirty_;

			inline unsigned char encode0() const;
			inline unsigned char encode1() const;
			inline unsigned char encode2() const;
			inline unsigned char encode3() const;

			base64Layer(const base64Layer &) = delete;
			void operator=(const base64Layer &) = delete;

			//- Add a character to the group, outputting when the group is full.
			void add(char c);

			//- The encoded length has 4 bytes out for every 3 bytes in.
			static std::size_t encodedLength(std::size_t n);

		public:
			base64Layer(std::ostream &os);
			~base64Layer();

			//- Encode the character sequence, writing when possible.
			void write(const char *s, std::streamsize n);

			inline void write(const uint64_t v) { write(reinterpret_cast<const char *>(&v), sizeof(uint64_t)); }
			inline void write(const int64_t v) { write(reinterpret_cast<const char *>(&v), sizeof(int64_t)); }
			inline void write(const int8_t v) { write(reinterpret_cast<const char *>(&v), sizeof(int8_t)); }
			inline void write(const uint8_t v) { write(reinterpret_cast<const char *>(&v), sizeof(uint8_t)); }

			inline void write(const int64_t *v, const int n) { write(reinterpret_cast<const char *>(v), n * sizeof(int64_t)); }
			inline void write(const double *v, const int n) { write(reinterpret_cast<const char *>(v), n * sizeof(double)); }
			inline void write(const float *v, const int n) { write(reinterpret_cast<const char *>(v), n * sizeof(float)); }

			//- Restart a new encoding sequence.
			void reset();

			//- End the encoding sequence, padding the final characters with '='.
			//  Return false if no encoding was actually performed.
			bool close();
		};
	} // namespace utils
} // namespace polyfem
