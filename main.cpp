#include <iostream>
#include <string>
#include "Eigen/Dense"
#include "Eigen/SparseLU"
#include "OpenMesh/Core/IO/MeshIO.hh"
#include "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"
#include "OpenMesh/Core/Geometry/EigenVectorT.hh"

using namespace std;
using namespace Eigen;

#define status int
#define OK 0

struct EigenTraits : OpenMesh::DefaultTraits {
    using Point = Eigen::Vector3d;
    using Normal = Eigen::Vector3d;
    using TexCoord2D = Eigen::Vector2d;
};

using EigenTriMesh = OpenMesh::TriMesh_ArrayKernelT<EigenTraits>;

status read_mesh(string path, EigenTriMesh &mesh);
status write_mesh(string path, EigenTriMesh &mesh, MatrixXd &x);
Vector3d cal_v4(Vector3d *v1, Vector3d *v2, Vector3d *v3);
status get_vertex_position(Vector3d v[],EigenTriMesh* mesh,unsigned int face_idx);
status solve_equation(MatrixXd *c, SparseMatrix<double> *A, MatrixXd *x);
status get_c( EigenTriMesh* s, EigenTriMesh* t, MatrixXd* c);
status get_A(EigenTriMesh* s,SparseMatrix<double>* A);

int main(int argc, char **argv) {

    /*read mesh from .obj*/
    EigenTriMesh s0,s1,t0,t1;
    read_mesh(argv[1],s0);
    read_mesh(argv[2],s1);
    read_mesh(argv[3],t0);
    read_mesh(argv[2],t1);

    SparseMatrix<double> A(3*s1.n_faces(),s1.n_vertices()+s1.n_faces());
    MatrixXd c(3*s1.n_faces(),3),x(s1.n_faces()+s1.n_vertices(),3);
    get_c(&s0,&t0,&c);
    get_A(&s1,&A);
    solve_equation(&c,&A,&x);
    write_mesh(argv[4],t1,x);
    return 0;
}

status read_mesh(string path, EigenTriMesh &mesh){
    if (!OpenMesh::IO::read_mesh(mesh, path))
    {
        std::cerr << "read error\n";
        exit(1);
    }
    return OK;
}

status write_mesh(string path, EigenTriMesh &mesh, MatrixXd &x){
    for(unsigned int i=0;i<mesh.n_vertices();++i){
        OpenMesh::VertexHandle vh = mesh.vertex_handle(i);
        for(int j=0;j<3;++j)
            mesh.point(vh)(j,0) = x(i,j);
    }
    if (!OpenMesh::IO::write_mesh(mesh, path))
    {
        std::cerr << "write error\n";
        exit(1);
    }
    return OK;
}

status get_c( EigenTriMesh* s, EigenTriMesh* t, MatrixXd* c){
    Vector3d vs0[4],vt0[4];
    for(unsigned int i=0;i<(*s).n_faces();++i){
        get_vertex_position(vs0,s,i);
        get_vertex_position(vt0,t,i);
        Matrix3d Vs,Vt,Q;
        Vs << vs0[1]-vs0[0],vs0[2]-vs0[0],vs0[3]-vs0[0];
        Vt << vt0[1]-vt0[0],vt0[2]-vt0[0],vt0[3]-vt0[0];
        Q = Vt*(Vs.inverse());
        for(int m=0;m<3;++m)
            for(int n=0;n<3;++n)
                (*c)(3*i+m,n) = Q(n,m);//对每个c转置
    }
    return OK;
}

status get_A(EigenTriMesh* s,SparseMatrix<double>* A){
    Vector3d vs1[4];
    for(unsigned int i=0;i<(*s).n_faces();++i){
        get_vertex_position(vs1,s,i);
        Matrix3d Vs,Vs_inverse;
        MatrixXd a(3,4);
        Vector3d vec;
        Vs << vs1[1]-vs1[0],vs1[2]-vs1[0],vs1[3]-vs1[0];
        Vs_inverse = Vs.inverse();
        vec << -Vs_inverse.block(0,0,3,1).sum(),-Vs_inverse.block(0,1,3,1).sum(),-Vs_inverse.block(0,2,3,1).sum();
        a << vec,Vs_inverse.transpose();
        OpenMesh::FaceHandle fh = (*s).face_handle(i);
        EigenTriMesh::FaceVertexIter fv_it = (*s).fv_iter(fh);
        for(int m=0; fv_it.is_valid(); ++fv_it,++m)
            for(int j=0;j<3;++j)
                (*A).insert(3*i+j,(*fv_it).idx()) = a(j,m);
        for(int j=0;j<3;++j)
            (*A).insert(3*i+j,(*s).n_vertices()+i) = a(j,3);
    }
    return OK;
}

status get_vertex_position(Vector3d v[],EigenTriMesh* mesh,unsigned int face_idx){
    OpenMesh::FaceHandle fh = (*mesh).face_handle(face_idx);
    EigenTriMesh::FaceVertexIter fv_it = (*mesh).fv_iter(fh);
    for(int i=0; fv_it.is_valid(); ++fv_it,++i) {
        v[i] = (*mesh).point(*fv_it);
    }
    v[3] = cal_v4(&v[0],&v[1],&v[2]);
    return OK;
}

Vector3d cal_v4(Vector3d *v1, Vector3d *v2, Vector3d *v3){
    return (*v1) + (((*v2)-(*v1)).cross((*v3)-(*v1))/sqrt(((*v2)-(*v1)).cross((*v3)-(*v1)).norm()));
}

status solve_equation(MatrixXd *c, SparseMatrix<double> *A, MatrixXd *x){
    SparseMatrix<double> left = ((*A).transpose()*(*A));
    SparseMatrix<double> right = (*A).transpose()*((*c).sparseView());
    SparseLU<SparseMatrix<double> > solver;
    solver.compute(left);
    if (solver.info() != Success){
        cout << (*A).transpose()*(*A) << endl <<endl;
        cout << solver.lastErrorMessage() << endl <<endl;
        std::cerr << "compute failure\n";
        exit(1);
    }
    (*x) = solver.solve(right);
    if (solver.info() != Success){
        std::cerr << "solve failure\n";
        exit(1);
    }
    cout << *x;
    return 0;
}
