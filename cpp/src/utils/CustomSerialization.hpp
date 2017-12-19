#ifndef CUSTOM_SERIALIZATION_HPP
#define CUSTOM_SERIALIZATION_HPP

#include "Problem.hpp"
#include "ElementBases.hpp"
#include "ElementAssemblyValues.hpp"
#include "LocalBoundary.hpp"
#include "Mesh2D.hpp"
#include "Mesh3D.hpp"

#include <igl/serialize.h>

namespace igl
{
    namespace serialization
    {
        //////////////poly_fem::Problem
        template <>
        inline void serialize(const poly_fem::Problem &obj,std::vector<char>& buffer)
        {
            ::igl::serialize(obj.problem_num(),std::string("problem_num"),buffer);
        }

        template <>
        inline void deserialize(poly_fem::Problem &obj,const std::vector<char>& buffer)
        {
            int val;
            ::igl::deserialize(val,std::string("problem_num"),buffer);
            obj.set_problem_num(val);
        }
        ///////////////////////////////

        //////////////poly_fem::ElementBases
        template <>
        inline void serialize(const poly_fem::ElementBases &obj,std::vector<char>& buffer)
        {
            ::igl::serialize(obj.bases, std::string("bases"), buffer);
            ::igl::serialize(obj.quadrature, std::string("quadrature"), buffer);
            ::igl::serialize(obj.has_parameterization, std::string("has_parameterization"), buffer);
        }

        template <>
        inline void deserialize(poly_fem::ElementBases &obj,const std::vector<char>& buffer)
        {
            ::igl::deserialize(obj.bases, std::string("bases"), buffer);
            ::igl::deserialize(obj.quadrature, std::string("quadrature"), buffer);
            ::igl::deserialize(obj.has_parameterization, std::string("has_parameterization"), buffer);
        }
        ///////////////////////////////

        //////////////poly_fem::Basis
        template <>
        inline void serialize(const poly_fem::Basis &obj,std::vector<char>& buffer)
        {
            //TODO
        }

        template <>
        inline void deserialize(poly_fem::Basis &obj,const std::vector<char>& buffer)
        {
            //TODO
        }
        ///////////////////////////////

        //////////////poly_fem::ElementAssemblyValues
        template <>
        inline void serialize(const poly_fem::ElementAssemblyValues &obj,std::vector<char>& buffer)
        {
            //TODO
        }

        template <>
        inline void deserialize(poly_fem::ElementAssemblyValues &obj,const std::vector<char>& buffer)
        {
            //TODO
        }
        ///////////////////////////////

        //////////////poly_fem::LocalBoundary
        template <>
        inline void serialize(const poly_fem::LocalBoundary &obj,std::vector<char>& buffer)
        {
            //TODO
        }

        template <>
        inline void deserialize(poly_fem::LocalBoundary &obj,const std::vector<char>& buffer)
        {
            //TODO
        }
        ///////////////////////////////

        //////////////poly_fem::Mesh2D
        template <>
        inline void serialize(const poly_fem::Mesh2D &obj,std::vector<char>& buffer)
        {
            //TODO
        }

        template <>
        inline void deserialize(poly_fem::Mesh2D &obj,const std::vector<char>& buffer)
        {
            //TODO
        }
        ///////////////////////////////

                //////////////poly_fem::Mesh3D
        template <>
        inline void serialize(const poly_fem::Mesh3D &obj,std::vector<char>& buffer)
        {
            //TODO
        }

        template <>
        inline void deserialize(poly_fem::Mesh3D &obj,const std::vector<char>& buffer)
        {
            //TODO
        }
        ///////////////////////////////
    }
}

#endif

