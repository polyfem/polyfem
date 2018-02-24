////////////////////////////////////////////////////////////////////////////////
#include <catch.hpp>
////////////////////////////////////////////////////////////////////////////////

TEST_CASE("foo", "[asd]") {
	auto fact = [] (int n) {
		int x = 1;
		for (int i = 1; i <= n; ++i) {
			x *= i;
		}
		return x;
	};

	REQUIRE(fact(2) == 2);
	REQUIRE(fact(3) == 6);
	REQUIRE(fact(4) == 24);
}

TEST_CASE("bar", "[asd]") {
	auto sum = [] (int n) {
		int x = 0;
		for (int i = 0; i <= n; ++i) {
			x += i;
		}
		return x;
	};

	REQUIRE(sum(3) == 6);
	REQUIRE(sum(4) == 10);
	REQUIRE(sum(5) == 15);
}
