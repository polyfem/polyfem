#include <polyfem/basis/ElementBases.hpp>

namespace polyfem::utils
{
    class Tree
    {
    public:
        Tree() {}
        Tree(const Tree &T) {
            if (T.has_children())
            {
                for (int i = 0; i < T.n_children(); i++)
                    this->children.push_back(std::make_unique<Tree>(T.child(i)));
            }
        }
        Tree operator=(const Tree &T) {
            if (this != &T)
            {
                this->children.clear();
                if (T.has_children())
                {
                    for (int i = 0; i < T.n_children(); i++)
                        this->children.push_back(std::make_unique<Tree>(T.child(i)));
                }
            }
            return *this;
        }
        void merge(const Tree &T) {
            if (T.has_children())
            {
                if (!this->has_children())
                    this->add_children(T.n_children());
                for (int i = 0; i < T.n_children(); i++)
                    this->child(i).merge(T.child(i));
            }
        }
        bool has_children() const { return !children.empty(); }
        int n_children() const { return children.size(); }
        Tree& child(int i) { return *children[i]; }
        const Tree& child(int i) const { return *children[i]; }
        void add_children(int n) {
            for (int i = 0; i < n; i++)
                children.push_back(std::make_unique<Tree>());
        }

    private:
        std::vector<std::unique_ptr<Tree>> children;
    };

    std::tuple<bool, int, Tree> isValid(
        const int dim,
        const std::vector<basis::ElementBases> &bases, 
        const Eigen::VectorXd &u);

    template<unsigned int nVar>
    bool isValid(const Eigen::Matrix<double, -1, nVar> &cp, int order);

    bool isValid(
        const int dim,
        const std::vector<basis::ElementBases> &bases, 
        const Eigen::VectorXd &u1,
        const Eigen::VectorXd &u2);

    double maxTimeStep(
        const int dim,
        const std::vector<basis::ElementBases> &bases, 
        const Eigen::VectorXd &u1,
        const Eigen::VectorXd &u2,
        double precision);
}
