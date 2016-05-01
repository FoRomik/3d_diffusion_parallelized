//Code for 3D heat conduction using flux based approach
#include<stdio.h>
#include<iostream>
#include<mpi.h>
using namespace std;
#define NP 16
	int i,ii,ij,ik,j,k,imax,jmax,kmax,Kmax;//have used two indices; imax,jmax for flux and Imax and Jmax for Temperature for convenience
	double dx,dy,dz,Lz; //grid properties
	int Imax=66; //Max grid points
	int Jmax=66;
	int Kmaxt=258;
	int count;
	double Lx=1.0;
	double Ly=1.0;
	double Lzt=1.0;
	int iter=0; //To keep track of iteration
	double rho=7750; //Physical parameters
	double cp=500;
	double kk=16.2;
	double qvolx=0; //Volumetric heat generation in x and y direction
	double qvoly=0;
	double qvolz=0;
	double Qgen;
	double unsteadi=1;
	MPI_Datatype layer;
	MPI_Status status;
	double ***x,***y,***z,***xx,***xy,***xz,***Qxold,***Qyold,***Qzold;
	double ***qx,***qy,***qz,***T,***q,***qold,***Told,***unstead;
	double dt=0.25;
	double alpha=kk/(rho*cp); //Courant number
	FILE *fp,*fp1;
	void MARRAY(void);
	void DISTRIBUTE_CELLS(int);
	double PRINT_DOMAIN(int);
	void SET_GEOMETRY(int);
	void writefi(int);
	void APPLYIC(int);
	void APPLYBC(int);
	void UPDATE(int);
	void CALC_FLUX(int);
	double EXCHANGE_CELLS(int);
	void CALC_TEMPERATURE(int);
	double CALC_UNSTEADY(int,int);
	void write_time(int,double,double,double);
	void writefile(int,double,double,double);
	void writefil(int);
	void write_total(int,double);
	void write_totall(int,double);
	MPI_Request request;
	double writef(int);
	FILE *f2;
int main(int argc, char *argv[])
{
	int size,rank;
	double unsteadmax=0,tot_main;
	time_t tt_start_main,tt_end_main;
	clock_t t1, t2,t3,t4;
	double RUN_TIME = 0;
	double COMM_TIME = 0;
	double PRINT_TIME=0;
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Type_vector(Imax*Jmax, 1, 1, MPI_DOUBLE, &layer);
	MPI_Type_commit(&layer);
	char *fname1="final.dat";
	char *fname2="initial.dat";
	fp=fopen(fname1,"w");
	fp1=fopen(fname2,"w");
	time (&tt_start_main);
	t1 = clock();
	DISTRIBUTE_CELLS(rank);
	imax=Imax-1; //To denote the flux indices
	jmax=Jmax-1;
	kmax=Kmax-1;
	MARRAY();
	SET_GEOMETRY(rank);
	APPLYIC(rank);
	//Calculating volumetric heat generation
	Qgen=(qvolx*dx)+(qvoly*dy)+(qvolz*dz);
	//Iteration begins here
	while(iter<=50000)
	{
	iter+=1;
	unsteadi=0;
	APPLYBC(rank);
	if(iter==1)
	{
		PRINT_TIME+=PRINT_DOMAIN(rank);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	COMM_TIME+=EXCHANGE_CELLS(rank);
	UPDATE(rank);
	CALC_FLUX(rank);
	CALC_TEMPERATURE(rank);
//	printf("\n Iteration : %d and unsteadiness = %f",iter,unsteadmax);
	}
	t2=clock();
	RUN_TIME += (double)((t2-t1)*1.0/CLOCKS_PER_SEC);
	PRINT_TIME+=writef(rank);
	write_time(rank,COMM_TIME,RUN_TIME,PRINT_TIME);
	t3=clock();
	time (&tt_start_main);
//	printf("\n Total run time in main is : %0.6f",difftime(tt_end_main, tt_start_main));
//	cout<<"\n Total run time in main is:"<<difftime(tt_end_main,tt_start_main)*1.0/CLOCKS_PER_SEC;
	tot_main=(double)((t3-t1)*1.0/CLOCKS_PER_SEC);
	write_total(rank,tot_main);
	MPI_Finalize();
	return 0;
}
void write_total(int rank, double tot_main)
{
	int write_done;
	while(1)
	{
		write_done = 0;

		MPI_Barrier(MPI_COMM_WORLD);
		
		if(rank==0)
		{
			write_totall(rank,tot_main);
			write_done = 1;
			MPI_Send(&write_done, 1, MPI_INT, rank+1, 111, MPI_COMM_WORLD);
			break;
		}
		else
		{
			MPI_Recv(&write_done, 1, MPI_INT, rank-1, 111, MPI_COMM_WORLD, &status);
			if(write_done==1)
			{
				write_totall(rank,tot_main);
				if(rank<(NP-1))
				MPI_Send(&write_done, 1, MPI_INT, rank+1, 111, MPI_COMM_WORLD);
				break;
			}
		}
	}MPI_Barrier(MPI_COMM_WORLD);
}
void write_totall(int rank, double tot_main)
{
	f2=fopen("time.txt","a+");
	fprintf(f2,"\n Total run time in main for rank %d is : %0.6f",rank,tot_main);
	fclose(f2);
}
void write_time(int rank,double COMM_TIME,double RUN_TIME,double PRINT_TIME)
{
	int write_done;
	while(1)
	{
		write_done = 0;

		MPI_Barrier(MPI_COMM_WORLD);
		
		if(rank==0)
		{
			writefile(rank,COMM_TIME,RUN_TIME,PRINT_TIME);
			write_done = 1;
			MPI_Send(&write_done, 1, MPI_INT, rank+1, 111, MPI_COMM_WORLD);
			break;
		}
		else
		{
			MPI_Recv(&write_done, 1, MPI_INT, rank-1, 111, MPI_COMM_WORLD, &status);
			if(write_done==1)
			{
				writefile(rank,COMM_TIME,RUN_TIME,PRINT_TIME);
				if(rank<(NP-1))
					MPI_Send(&write_done, 1, MPI_INT, rank+1, 111, MPI_COMM_WORLD);
				break;
			}
		}
	}MPI_Barrier(MPI_COMM_WORLD);
}
void writefile(int rank,double COMM_TIME,double RUN_TIME,double PRINT_TIME)
{
	switch(rank)
		{
			case 0:
			f2=fopen("time.txt","w");	
			fprintf(f2,"\n The communication time for rank:%d is %0.6f",rank,COMM_TIME);
			fprintf(f2,"\n The run time for rank:%d is %0.6f",rank,RUN_TIME);
			fprintf(f2,"\n The file writing time for rank:%d is %0.6f\n",rank,PRINT_TIME);
			fclose(f2);
			break;
			case (NP-1):
			f2=fopen("time.txt","a+");	
			fprintf(f2,"\n The communication time for rank:%d is %0.6f",rank,COMM_TIME);
			fprintf(f2,"\n The run time for rank:%d is %0.6f",rank,RUN_TIME);
			fprintf(f2,"\n The file writing time for rank:%d is %0.6f\n",rank,PRINT_TIME);
			fclose(f2);
			break;
			default:
			f2=fopen("time.txt","a+");	
			fprintf(f2,"\n The communication time for rank:%d is %0.6f",rank,COMM_TIME);
			fprintf(f2,"\n The run time for rank:%d is %0.6f",rank,RUN_TIME);
			fprintf(f2,"\n The file writing time for rank:%d is %0.6f\n",rank,PRINT_TIME);
			fclose(f2);
			break;
		}
}
double writef(int rank)
{
	clock_t t1,t2;
	t1=clock();
	int write_done;
	while(1)
	{
		write_done = 0;

		MPI_Barrier(MPI_COMM_WORLD);
		
		if(rank==0)
		{
			writefi(rank);
			write_done = 1;
			MPI_Send(&write_done, 1, MPI_INT, rank+1, 111, MPI_COMM_WORLD);
			break;
		}
		else
		{
			MPI_Recv(&write_done, 1, MPI_INT, rank-1, 111, MPI_COMM_WORLD, &status);
			if(write_done==1)
			{
				writefi(rank);
				if(rank<(NP-1))
					MPI_Send(&write_done, 1, MPI_INT, rank+1, 111, MPI_COMM_WORLD);
				break;
			}
		}
	}MPI_Barrier(MPI_COMM_WORLD);
	t2=clock();
	return (double)((t2-t1)*1.0/CLOCKS_PER_SEC);
}
void writefi(int rank)
{		
		FILE *f1;
		switch(rank)
		{
			case 0:
			f1=fopen("final.dat","w");	
			fprintf(f1,"VARIABLES = \"Z\", \"X\", \"Y\", \"T\"\n");
			fprintf(f1,"ZONE I=%d, J=%d, K=%d, F=POINT",Imax,Jmax,Kmaxt);	
			for(k=0;k<Kmax-1;k++)
			for(i=0;i<Imax;i++)
			for(j=0; j<Jmax; j++)
			fprintf(f1,"\n%0.6f %0.6f %0.6f %0.6f",z[k][i][j],x[k][i][j],y[k][i][j],T[k][i][j]);
			fclose(f1);	
			break;
			case (NP-1):
			f1=fopen("final.dat","a+");	
			for(k=1;k<Kmax;k++)
			for(i=0;i<Imax;i++)
			for(j=0; j<Jmax; j++)
			fprintf(f1,"\n%0.6f %0.6f %0.6f %0.6f",z[k][i][j],x[k][i][j],y[k][i][j],T[k][i][j]);
			fclose(f1);
			break;
			default:
			f1=fopen("final.dat","a+");	
			for(k=1;k<Kmax-1;k++)
			for(i=0;i<Imax;i++)
			for(j=0; j<Jmax; j++)
			fprintf(f1,"\n%0.6f %0.6f %0.6f %0.6f",z[k][i][j],x[k][i][j],y[k][i][j],T[k][i][j]);
			fclose(f1);	
		}
}

void MARRAY()
{
		//Domain
	x=new double**[Kmax];
	y=new double**[Kmax];
	z=new double**[Kmax];
	xx=new double**[kmax];
	xy=new double**[kmax];
	xz=new double**[kmax];
	Qxold=new double**[kmax];
	Qyold=new double**[kmax];
	Qzold=new double**[kmax];
	qold=new double**[kmax];
	q=new double**[kmax];
	unstead=new double**[kmax];
	for(k=0;k<Kmax;k++)
	{
		x[k]=new double*[Jmax];
		y[k]=new double*[Jmax];
		z[k]=new double*[Jmax];
		for(j=0;j<Jmax;j++)
		{
			x[k][j]=new double[Imax];
			y[k][j]=new double[Imax];
			z[k][j]=new double[Imax];
			
		}
	}
	for(k=0;k<kmax;k++)
	{
		xx[k]=new double*[jmax];
		xy[k]=new double*[jmax];
		xz[k]=new double*[jmax];
		Qxold[k]=new double*[jmax];
		Qyold[k]=new double*[jmax];
		Qzold[k]=new double*[jmax];
		qold[k]=new double*[jmax];
		q[k]=new double*[jmax];
		unstead[k]=new double*[jmax];
		for(j=0;j<jmax;j++)
		{
			xx[k][j]=new double[imax];
			xy[k][j]=new double[imax];
			xz[k][j]=new double[imax];
			Qxold[k][j]=new double[imax];
			Qyold[k][j]=new double[imax];
			Qzold[k][j]=new double[imax];
			qold[k][j]=new double[imax];
			q[k][j]=new double[imax];
			unstead[k][j]=new double[imax];
		}
	}
	
	T = new double **[Kmax];
	double **Ti = new double *[Jmax*Kmax];
	double *Tii = new double[Imax*Jmax*Kmax];
	for(k=0; k<Kmax; k++, Ti += Jmax)
	{
		T[k] = Ti;
		for(j=0; j<Jmax; j++, Tii += Imax)
			T[k][j] = Tii;
	}
	Told = new double **[Kmax];
	double **Toldi = new double *[Jmax*Kmax];
	double *Toldii = new double[Imax*Jmax*Kmax];
	for(k=0; k<Kmax; k++, Toldi += Jmax)
	{
		Told[k] = Toldi;
		for(j=0; j<Jmax; j++, Toldii += Imax)
			Told[k][j] = Toldii;
	}
	qx = new double **[kmax];
	double **qxi = new double *[kmax*jmax];
	double *qxii = new double[imax*jmax*kmax];
	for(k=0; k<kmax; k++, qxi += jmax)
	{
		qx[k] = qxi;
		for(j=0; j<jmax; j++, qxii += imax)
			qx[k][j] = qxii;
	}
	qy = new double **[kmax];
	double **qyi = new double *[jmax*kmax];
	double *qyii = new double[imax*jmax*kmax];
	for(k=0; k<kmax; k++, qyi += jmax)
	{
		qy[k] = qyi;
		for(j=0; j<jmax; j++, qyii += imax)
			qy[k][j] = qyii;
	}
	qz = new double **[kmax];
	double **qzi = new double *[jmax*kmax];
	double *qzii = new double[imax*jmax*kmax];
	for(k=0; k<kmax; k++, qzi += jmax)
	{
		qz[k] = qzi;
		for(j=0; j<jmax; j++, qzii += imax)
			qz[k][j] = qzii;
	}
}
void DISTRIBUTE_CELLS(int rank)
{
	if(((Kmaxt-2)%NP) != 0)
	{
		if(rank==0)
		{	cout<<endl<<"Incrementing no. of cells in longitudinal direction to achieve equal distribution of cells within processors"<<endl;
			cout<<"Input no. of cells = "<<Kmaxt<<endl;	}
		Kmaxt = Kmaxt + NP - ((Kmaxt-2)%NP);
		if(rank==0)
			cout<<"Modified no. of cells = "<<Kmaxt<<endl<<endl;
	}
	// Number of real cells per processor array
	Kmax = (Kmaxt-2)/NP;
	// Total no. of grid points required per array per processor
	Kmax = Kmax+2;
}
double PRINT_DOMAIN(int rank)
{
	clock_t t1,t2;
	t1=clock();
	int write_done;
	while(1)
	{
		write_done = 0;

		MPI_Barrier(MPI_COMM_WORLD);
		
		if(rank==0)
		{
			writefil(rank);
			write_done = 1;
			MPI_Send(&write_done, 1, MPI_INT, rank+1, 111, MPI_COMM_WORLD);
			break;
		}

		else
		{
			MPI_Recv(&write_done, 1, MPI_INT, rank-1, 111, MPI_COMM_WORLD, &status);
			if(write_done==1)
			{
				writefil(rank);
				if(rank<(NP-1))
					MPI_Send(&write_done, 1, MPI_INT, rank+1, 111, MPI_COMM_WORLD);
				break;
			}
		}
	}MPI_Barrier(MPI_COMM_WORLD);
	t2=clock();
	return (double)((t2-t1)*1.0/CLOCKS_PER_SEC);
}
void writefil(int rank)
{		
		FILE *f3;
		switch(rank)
		{
			case 0:
			f3=fopen("initial.dat","w");
			fprintf(f3,"VARIABLES = \"Z\", \"X\", \"Y\", \"T\"\n");
			fprintf(f3,"ZONE I=%d, J=%d, K=%d, F=POINT",Imax,Jmax,Kmaxt);
			for(k=0;k<Kmax-1;k++)
			for(i=0;i<Imax;i++)
			for(j=0; j<Jmax; j++)
			fprintf(f3,"\n%0.6f %0.6f %0.6f %0.6f",z[k][i][j],x[k][i][j],y[k][i][j],T[k][i][j]);
			fclose(f3);	
			break;
			case (NP-1):
			f3=fopen("initial.dat","a+");	
			for(k=1;k<Kmax;k++)
			for(i=0;i<Imax;i++)
			for(j=0; j<Jmax; j++)
			fprintf(f3,"\n%0.6f %0.6f %0.6f %0.6f",z[k][i][j],x[k][i][j],y[k][i][j],T[k][i][j]);
			fclose(f3);
			break;
			default:
			f3=fopen("initial.dat","a+");	
			for(k=1;k<Kmax-1;k++)
			for(i=0;i<Imax;i++)
			for(j=0; j<Jmax; j++)
			fprintf(f3,"\n%0.6f %0.6f %0.6f %0.6f",z[k][i][j],x[k][i][j],y[k][i][j],T[k][i][j]);
			fclose(f3);	
		}
}
void SET_GEOMETRY(int rank)
{
	dx=Lx/(imax-1);
	dy=Ly/(jmax-1);
	dz=Lzt/(Kmaxt-2);
	Lz=Lzt/NP;
	//Defining coordinates for flux
	for(k=0;k<kmax;k++)
	{
		for(i=0;i<imax;i++)
	{
		for(j=0;j<jmax;j++)
		{
				xx[k][i][j]=j*dx;
				xy[k][i][j]=i*dy;
				xz[k][i][j]=k*dz +(rank*Lz);
//				printf("\n rank=%d xz[%d][%d][%d]=%f",rank,k,i,j,xz[k][i][j]);
		}
	}	
	}

	//Defining coordinates for temperature
	for(k=1;k<kmax;k++)
	{
		for(i=1;i<imax;i++)
	{
		for(j=1;j<jmax;j++)
		{
				x[k][i][j]=(xx[k][i][j-1]+xx[k][i][j]+xx[k][i-1][j-1]+xx[k][i-1][j]+xx[k-1][i-1][j]+xx[k-1][i-1][j-1]+xx[k-1][i][j]+xx[k-1][i][j-1])/8;
				y[k][i][j]=(xy[k][i][j-1]+xy[k][i][j]+xy[k][i-1][j-1]+xy[k][i-1][j]+xy[k-1][i-1][j]+xy[k-1][i-1][j-1]+xy[k-1][i][j]+xy[k-1][i][j-1])/8;
				z[k][i][j]=(xz[k][i][j-1]+xz[k][i][j]+xz[k][i-1][j-1]+xz[k][i-1][j]+xz[k-1][i-1][j]+xz[k-1][i-1][j-1]+xz[k-1][i][j]+xz[k-1][i][j-1])/8;
			
//			printf("\n rank=%d x[%d][%d][%d]=%f",rank,k,i,j,z[k][i][j]);
		}
	}	
	}
	switch(rank)
	{
		case 0:
			for(j=1;j<jmax;j++)
		{
			for(i=1;i<imax;i++)
			{
				x[0][i][j]=x[1][i][j];
				y[0][i][j]=y[1][i][j];
				z[0][i][j]=0;
//					printf("\n rank=%d x[%d][%d][%d]=%f",rank,k,i,j,x[0][i][j]);
//					printf("\n rank=%d y[%d][%d][%d]=%f",rank,k,i,j,y[0][i][j]);
//					printf("\n rank=%d z[%d][%d][%d]=%f",rank,k,i,j,z[0][i][j]);
			}
		}
		for(j=1;j<jmax;j++)
		{
			for(i=1;i<imax;i++)
			{
				x[0][i][0]=0;
				y[0][i][0]=y[0][i][1];
				z[0][i][0]=z[0][i][1];
				x[0][0][j]=x[0][1][j];
				z[0][0][j]=z[0][1][j];
				y[0][0][j]=0;
				x[0][i][Jmax-1]=Lx;
				y[0][i][Jmax-1]=y[0][i][Jmax-2];
				z[0][i][Jmax-1]=z[0][i][Jmax-2];
				x[0][Imax-1][j]=x[0][Imax-2][j];
				z[0][Imax-1][j]=z[0][Imax-2][j];
				y[0][Imax-1][j]=Ly;
//					printf("\n rank=%d y[%d][%d][%d]=%f",rank,k,i,j,y[0][i][j]);
			}
		}
				x[0][0][0]=0;
				y[0][0][0]=0;
				z[0][0][0]=0;
				x[0][Imax-1][0]=0;
				y[0][Imax-1][0]=Ly;
				z[0][Imax-1][0]=0;
				x[0][0][Jmax-1]=Lx;
				y[0][0][Jmax-1]=0;
				z[0][0][Jmax-1]=0;
				x[0][Imax-1][Jmax-1]=Lx;
				y[0][Imax-1][Jmax-1]=Ly;
				z[0][Imax-1][Jmax-1]=0;
		for(k=1;k<kmax;k++)
		{
			for(j=0;j<Jmax;j++)
			{
				for(i=0;i<Imax;i++)
				{
				if(k==1)
				{
				x[k][i][j]=x[k-1][i][j];
				y[k][i][j]=y[k-1][i][j];
				z[k][i][j]=z[k-1][i][j]+(dz/2);	
				}	
				else
				{
					x[k][i][j]=x[k-1][i][j];
					y[k][i][j]=y[k-1][i][j];
					z[k][i][j]=z[k-1][i][j]+dz;
				}
				}
			}
		}
		break;
		case (NP-1):
			for(j=1;j<jmax;j++)
		{
			for(i=1;i<imax;i++)
			{
				x[Kmax-1][i][j]=x[Kmax-2][i][j];
				y[Kmax-1][i][j]=y[Kmax-2][i][j];
				z[Kmax-1][i][j]=Lzt;
//					printf("\n rank=%d x[%d][%d][%d]=%f",rank,k,i,j,x[0][i][j]);
//					printf("\n rank=%d y[%d][%d][%d]=%f",rank,k,i,j,y[0][i][j]);
//					printf("\n rank=%d z[%d][%d][%d]=%f",rank,k,i,j,z[0][i][j]);
			}
		}
		for(j=1;j<jmax;j++)
		{
			for(i=1;i<imax;i++)
			{
				x[Kmax-1][i][0]=0;
				y[Kmax-1][i][0]=y[Kmax-1][i][1];
				z[Kmax-1][i][0]=z[Kmax-1][i][1];
				x[Kmax-1][0][j]=x[Kmax-1][1][j];
				z[Kmax-1][0][j]=z[Kmax-1][1][j];
				y[Kmax-1][0][j]=0;
				x[Kmax-1][i][Jmax-1]=Lx;
				y[Kmax-1][i][Jmax-1]=y[Kmax-1][i][Jmax-2];
				z[Kmax-1][i][Jmax-1]=z[Kmax-1][i][Jmax-2];
				x[Kmax-1][Imax-1][j]=x[Kmax-1][Imax-2][j];
				z[Kmax-1][Imax-1][j]=z[Kmax-1][Imax-2][j];
				y[Kmax-1][Imax-1][j]=Ly;
//					printf("\n rank=%d y[%d][%d][%d]=%f",rank,k,i,j,y[0][i][j]);
			}
		}
				x[Kmax-1][0][0]=0;
				y[Kmax-1][0][0]=0;
				z[Kmax-1][0][0]=Lzt;
				x[Kmax-1][Imax-1][0]=0;
				y[Kmax-1][Imax-1][0]=Ly;
				z[Kmax-1][Imax-1][0]=Lzt;
				x[Kmax-1][0][Jmax-1]=Lx;
				y[Kmax-1][0][Jmax-1]=0;
				z[Kmax-1][0][Jmax-1]=Lzt;
				x[Kmax-1][Imax-1][Jmax-1]=Lx;
				y[Kmax-1][Imax-1][Jmax-1]=Ly;
				z[Kmax-1][Imax-1][Jmax-1]=Lzt;
		for(k=kmax-1;k>=1;k--)
		{
			for(j=Jmax-1;j>=0;j--)
			{
				for(i=Imax-1;i>=0;i--)
				{
				if(k==kmax-1)
				{
				x[k][i][j]=x[k+1][i][j];
				y[k][i][j]=y[k+1][i][j];
				z[k][i][j]=z[k+1][i][j]-(dz/2);	
				}	
				else
				{
					x[k][i][j]=x[k+1][i][j];
					y[k][i][j]=y[k+1][i][j];
					z[k][i][j]=z[k+1][i][j]-dz;
				}
				}
			}
		}
		break;
		default:
			for(j=1;j<jmax;j++)
		{
			for(i=1;i<imax;i++)
			{
				x[1][i][0]=0;
				y[1][i][0]=y[1][i][1];
				z[1][i][0]=z[1][i][1];
				x[1][0][j]=x[1][1][j];
				z[1][0][j]=z[1][1][j];
				y[1][0][j]=0;
				x[1][i][Jmax-1]=Lx;
				y[1][i][Jmax-1]=y[1][i][Jmax-2];
				z[1][i][Jmax-1]=z[1][i][Jmax-2];
				x[1][Imax-1][j]=x[1][Imax-2][j];
				z[1][Imax-1][j]=z[1][Imax-2][j];
				y[1][Imax-1][j]=Ly;
//					printf("\n rank=%d y[%d][%d][%d]=%f",rank,k,i,j,y[0][i][j]);
			}
		}
				x[1][0][0]=0;
				y[1][0][0]=0;
				z[1][0][0]=rank*Lz+(0.5*dz);
				x[1][Imax-1][0]=0;
				y[1][Imax-1][0]=Ly;
				z[1][Imax-1][0]=rank*Lz+(0.5*dz);
				x[1][0][Jmax-1]=Lx;
				y[1][0][Jmax-1]=0;
				z[1][0][Jmax-1]=rank*Lz+(0.5*dz);
				x[1][Imax-1][Jmax-1]=Lx;
				y[1][Imax-1][Jmax-1]=Ly;
				z[1][Imax-1][Jmax-1]=rank*Lz+(0.5*dz);
				for(k=2;k<kmax;k++)
		{
			for(j=0;j<Jmax;j++)
			{
				for(i=0;i<Imax;i++)
				{
					x[k][i][j]=x[k-1][i][j];
					y[k][i][j]=y[k-1][i][j];
					z[k][i][j]=z[k-1][i][j]+dz;
				}
			}
		}
		break;
	}
}
void APPLYIC(int rank)
{
	//Initial condition
	for(k=1;k<kmax;k++)
	{
		for(i=1;i<imax;i++)
	{
		for(j=1;j<jmax;j++)
		{
		T[k][i][j]=30;	
		}
	}
		
	}
}
void APPLYBC(int rank)
{
for(j=0;j<Jmax;j++)
{
	for(i=0;i<Imax;i++)
	{	
		if(rank==0)
		T[0][i][j]=100;
		else
		T[0][i][j]=0;
		if(rank==NP-1)
		T[Kmax-1][i][j]=300;
		else
		T[Kmax-1][i][j]=0;
	}
}
for(k=0;k<Kmax;k++)
{
	for(j=0;j<Jmax;j++)
	{
		T[k][0][j]=400;
		T[k][Imax-1][j]=200;
	}
}
for(k=0;k<Kmax;k++)
{
	for(i=0;i<Imax;i++)
	{
		T[k][i][0]=500;
		T[k][i][Jmax-1]=600;
	}
}
/*
for(k=0;k<kmax;k++)
{
	for(j=0;j<jmax;j++)
	{
		for(i=0;i<imax;i++)
		{
			printf("\n rank=%d T[%d][%d][%d]=%0.6f",rank,k,i,j,T[k][i][j]);
		}
	}
}*/
}
void UPDATE(int rank)
{
	for(k=0;k<Kmax;k++)
	{
	for(i=0;i<Imax;i++)
	{
		for(j=0;j<Jmax;j++)
		{
			Told[k][i][j]=T[k][i][j];
		}
	}
	}
}
void CALC_FLUX(int rank)
{
		//Calculating flux in x direction
	for(k=0;k<kmax;k++)
	{
	for(i=0;i<imax;i++)
	{
	for(j=0;j<jmax;j++)
	{
	if((j==0)||(j==jmax-1))
		qx[k][i][j]=-(2*kk)*((Told[k][i][j+1]-Told[k][i][j])/dx);
	else
		qx[k][i][j]=-kk*((Told[k][i][j+1]-Told[k][i][j])/dx);
		
	//	printf("\n qx[%d][%d]=%f",i,j,qx[i][j]);
	}
	}
	}
	//Caculating flux in y direction
	for(k=0;k<kmax;k++)
	{
	for(i=0;i<imax;i++)
	{
	for(j=0;j<jmax;j++)
	{
		if((i==0)||(i==imax-1))
		qy[k][i][j]=-(2*kk)*((Told[k][i+1][j]-Told[k][i][j])/dy);
		else
		qy[k][i][j]=-kk*((Told[k][i+1][j]-Told[k][i][j])/dy);
		
	//	printf("\n qy[%d][%d]=%f",i,j,qy[i][j]);
	}	
	}
	}
	//Caculating flux in z direction
	for(k=0;k<kmax;k++)
	{
	for(i=0;i<imax;i++)
	{
	for(j=0;j<jmax;j++)
	{
		if((k==0)||(k==kmax-1))
		qz[k][i][j]=-(2*kk)*((Told[k+1][i][j]-Told[k][i][j])/dz);
		else
		qz[k][i][j]=-kk*((Told[k+1][i][j]-Told[k][i][j])/dz);
		
	//	printf("\n qy[%d][%d]=%f",i,j,qy[i][j]);
	}
	}
	}
	//Calculating temperature from flux
	for(k=1;k<kmax;k++)
	{
	for(i=1;i<imax;i++)
	{
		for(j=1;j<jmax;j++)
		{
			Qxold[k][i][j]=qx[k][i][j-1]-qx[k][i][j];
		}
	}
	}
	for(k=1;k<kmax;k++)
	{
		for(i=1;i<imax;i++)
	{
		for(j=1;j<jmax;j++)
		{
			Qyold[k][i][j]=qy[k][i-1][j]-qy[k][i][j];
		}
	}
	}
		for(k=1;k<kmax;k++)
	{
		for(i=1;i<imax;i++)
	{
		for(j=1;j<jmax;j++)
		{
			Qzold[k][i][j]=qz[k-1][i][j]-qz[k][i][j];
		}
	}
	}
/*	for(k=1;k<kmax;k++)
	{
		for(i=1;i<imax;i++)
		{
			for(j=1;j<jmax;j++)
			{
				printf("\n rank=%d Qold[%d][%d][%d]=%0.6f",rank,k,i,j,Qzold[k][i][j]);
			}
		}
	}*/
}
double EXCHANGE_CELLS(int rank)
{
	clock_t t1, t2;
	t1 = clock();
	switch(rank)
	{
		case 0:
		MPI_Send(&T[kmax-1][0][0],1,layer,rank+1,1,MPI_COMM_WORLD);
		MPI_Recv(&T[Kmax-1][0][0],1,layer,rank+1,1,MPI_COMM_WORLD,&status);
//		MPI_Get_count(&status,MPI_DOUBLE,&count);
//		printf("\n rank=%d Received=%d",rank,count);
		break;
		case NP-1:
		MPI_Send(&T[1][0][0],1,layer,rank-1,1,MPI_COMM_WORLD);
		MPI_Recv(&T[0][0][0],1,layer,rank-1,1,MPI_COMM_WORLD,&status);
//		MPI_Get_count(&status,MPI_DOUBLE,&count);
//		printf("\n rank=%d Received=%d",rank,count);
		break;
		default:
		MPI_Send(&T[1][0][0],1,layer,rank+1,1,MPI_COMM_WORLD);
		MPI_Recv(&T[0][0][0],1,layer,rank-1,1,MPI_COMM_WORLD,&status);
		MPI_Send(&T[kmax-1][0][0],1,layer,rank-1,1,MPI_COMM_WORLD);
		MPI_Recv(&T[Kmax-1][0][0],1,layer,rank+1,1,MPI_COMM_WORLD,&status);
		break;
	}
	t2 = clock();
	return((double)((t2-t1)*1.0/CLOCKS_PER_SEC));
}
void CALC_TEMPERATURE(int rank)
{
	for(k=1;k<kmax;k++)
	{
	for(i=1;i<imax;i++)
	{
		for(j=1;j<jmax;j++)
		{
			T[k][i][j]=Told[k][i][j]+(dt/(rho*cp*dx))*(Qxold[k][i][j])+(dt/(rho*cp*dy))*(Qyold[k][i][j])+(Qzold[k][i][j])*(dt/(rho*cp*dz))+Qgen;
		//	printf("T[%d][%d]=%f \t",i,j,T[i][j]);	
		}
		//printf("\n");
	}
	}
}
double CALC_UNSTEADY(int rank,int size)
{
	double unsteady[size];
	double maximum;
	//Calculating unsteadiness - with only temperature difference
	for(k=0;k<Kmax;k++)
	{
	for(i=0;i<Imax;i++)
	{
		for(j=0;j<Jmax;j++)
		{
			unstead[k][i][j]=T[k][i][j]-Told[k][i][j];
			if(unstead[k][i][j]<0)
			{
				unstead[k][i][j]=-unstead[k][i][j];
			}
		//	printf("\n %d %d %f",i,j,unstead[i-1][j-1]);
		}
	}
	}
	for(k=0;k<Kmax;k++)
	{
	for(i=0;i<Imax;i++)
	{
		for(j=0;j<Jmax;j++)
		{
			if(unstead[k][i][j]>unsteadi)
			{
				unsteadi=unstead[k][i][j];
			//	printf("\n %f",unsteadi);
			}
		}
	}
	}
	if(rank!=0)
	{
		MPI_Send(&unsteadi,1,MPI_DOUBLE,0,10,MPI_COMM_WORLD);
	}
	else if(rank==0)
	{
		unsteady[0]=unsteadi;
		for(i=1;i<size;i++)
		MPI_Recv(&unsteady[i],1,MPI_DOUBLE,i,10,MPI_COMM_WORLD,&status);
		maximum = unsteady[0];
  		for (i=0;i<size;i++)
 		{
  		  if (unsteady[i] > maximum)
  		  {
    	   maximum  = unsteady[i];
  		  }
 		}
	}
	return maximum;
}
