package com.nogotech.ondeviceml.ui.home

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.databinding.DataBindingUtil
import androidx.fragment.app.Fragment
import androidx.lifecycle.Observer
import androidx.lifecycle.ViewModelProvider
import com.bumptech.glide.Glide
import com.nogotech.ondeviceml.R
import com.nogotech.ondeviceml.databinding.FragmentHomeBinding

class HomeFragment : Fragment() {

    private lateinit var homeViewModel: HomeViewModel
    private lateinit var binding: FragmentHomeBinding

    override fun onCreateView(
            inflater: LayoutInflater,
            container: ViewGroup?,
            savedInstanceState: Bundle?
    ): View? {
        binding = DataBindingUtil.inflate(inflater, R.layout.fragment_home, container, false)

        val application = requireNotNull(this.activity).application

        val glide = Glide.with(requireContext())

        val viewModelFactory = HomeViewModelFactory(glide, application)

        homeViewModel =
            ViewModelProvider(this, viewModelFactory).get(HomeViewModel::class.java)

        binding.homeViewModel = homeViewModel
        binding.lifecycleOwner = this

        homeViewModel.loadBitmap(R.drawable.img_aso)


        return binding.root
    }
}