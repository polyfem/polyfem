#pragma once

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
        bool merge(const Tree &T, int max_depth = 2) {
            bool flag = false;
            if (!T.has_children() || max_depth <= 0)
                return flag;
            if (!this->has_children())
            {
                this->add_children(T.n_children());
                flag = true;
                max_depth--;
            }
            for (int i = 0; i < T.n_children(); i++)
                flag = this->child(i).merge(T.child(i), max_depth) || flag;
            return flag;
        }

        // Debug print
        friend std::ostream& operator<<(
            std::ostream& ost, const Tree& T
        ) {
            ost << "(";
            if (T.has_children())
            {
                for (int i = 0; i < T.n_children(); i++)
                    ost << T.child(i) << ", ";
            }
            ost << ")";
            return ost;
        }

        bool has_children() const { return !children.empty(); }
        int n_children() const { return children.size(); }
        int depth() const {
            if (!has_children())
                return 0;
            int d = 0;
            for (int i = 0; i < n_children(); i++)
                d = std::max(d, child(i).depth());
            return d + 1;
        }
        int n_leaves() const {
            if (!has_children())
                return 1;
            int n = 0;
            for (int i = 0; i < n_children(); i++)
                n += child(i).n_leaves();
            return n;
        }
        Tree& child(int i) { return *children[i]; }
        const Tree& child(int i) const { return *children[i]; }
        void add_children(int n) {
            for (int i = 0; i < n; i++)
                children.push_back(std::make_unique<Tree>());
        }

    private:
        std::vector<std::unique_ptr<Tree>> children;
    };

    Eigen::VectorXd robust_evaluate_jacobian(
        const int order,
        const Eigen::MatrixXd &cp,
        const Eigen::MatrixXd &uv);

    std::vector<int> count_invalid(
        const int dim,
        const std::vector<basis::ElementBases> &bases, 
        const std::vector<basis::ElementBases> &gbases,
        const Eigen::VectorXd &u);

    std::tuple<bool, int, Tree> is_valid(
        const int dim,
        const std::vector<basis::ElementBases> &bases, 
        const std::vector<basis::ElementBases> &gbases,
        const Eigen::VectorXd &u,
        const double threshold = 0);

    bool is_valid(
        const int dim,
        const std::vector<basis::ElementBases> &bases, 
        const std::vector<basis::ElementBases> &gbases,
        const Eigen::VectorXd &u1,
        const Eigen::VectorXd &u2,
        const double threshold = 0);

    std::tuple<double, int, double, Tree> max_time_step(
        const int dim,
        const std::vector<basis::ElementBases> &bases, 
        const std::vector<basis::ElementBases> &gbases,
        const Eigen::VectorXd &u1,
        const Eigen::VectorXd &u2,
        double precision = .25);

    Eigen::MatrixXd extract_nodes(const int dim, const basis::ElementBases &basis, const basis::ElementBases &gbasis, const Eigen::VectorXd &u, int order);
    Eigen::MatrixXd extract_nodes(const int dim, const std::vector<basis::ElementBases> &bases, const std::vector<basis::ElementBases> &gbases, const Eigen::VectorXd &u, int order, int n_elem = -1);
}
